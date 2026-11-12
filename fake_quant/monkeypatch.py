# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import copy
from contextlib import contextmanager, ExitStack
import functools
import types
from typing import Callable, TypeVar
from collections import OrderedDict

import torch.nn as nn


def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__,
        globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(
    module, method_name, function_name, wrapper_fn
):
    """
    This function adds a wrapper after the output of a function call in the method named `method_name`.
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    """

    original_method = getattr(module, method_name).__func__
    if hasattr(original_method, "__wrapped__"):
        original_method = original_method.__wrapped__
    method_globals = dict(original_method.__globals__)
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper


class BreakpointException(Exception):
    pass


class BreakpointGroup:
    def __init__(self):
        self.breakpoints = {}

    def register(self, key):
        self.breakpoints[key] = False

    def unregister(self, key):
        del self.breakpoints[key]

    def raise_breakpoints(self, key, unregister=False):
        if unregister:
            self.unregister(key)
        else:
            self.breakpoints[key] = True

        if all(enable for enable in self.breakpoints.values()):
            # Reset breakpoints
            keys = list(self.breakpoints.keys())
            for key in keys:
                self.breakpoints[key] = False
            raise BreakpointException(f"All breakpoints reached. {keys}")


class IOCatcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        self.use_catched_output = False
        self.__catch_input = False
        self.__catch_output = False
        self.__once = True
        self.bp: BreakpointGroup | None = None

    def catch(self, input=False, output=False, once=True, bp=None):
        """Set catch input/output options of the module.

        Args:
            input (bool, optional): Whether to catch input. Defaults to False.
            output (bool, optional): Whether to catch output. Defaults to False.
            once (bool, optional): Whether to catch only once. Defaults to True.
            bp (BreakpointGroup, optional): Breakpoint group to use.
                - If None, no break after catching. Defaults to None.
        """
        self.__catch_input = input
        self.__catch_output = output
        self.__once = once

        if self.bp is not None:
            self.bp.unregister(self)
        self.bp = bp
        if self.bp is not None:
            self.bp.register(self)

    @contextmanager
    def catch_ctx(self, input=False, output=False, once=True, bp=None):
        self.catch(input=input, output=output, once=once, bp=bp)
        try:
            yield self
        finally:
            self.catch(input=False, output=False, once=True, bp=None)

    def __raise(self):
        if self.bp is not None:
            bp = self.bp
            if self.__once:
                self.bp = None
            bp.raise_breakpoints(self, unregister=self.__once)

    def forward(self, *args, **kwargs):
        if self.__catch_input:
            self.catched_input = (args, kwargs)
            if self.__once:
                self.__catch_input = False
            if not self.__catch_output:
                self.__raise()

        if self.__catch_output:
            output = self.module(*args, **kwargs)
            self.catched_output = output
            if self.__once:
                self.__catch_output = False
            self.__raise()
            return output

        if self.use_catched_output and self.catched_output is not None:
            return self.catched_output

        return self.module(*args, **kwargs)

    def clear_cache(self):
        del self.catched_input
        del self.catched_output


@contextmanager
def merge_ctx(*ctx_managers):
    with ExitStack() as stack:
        for ctx in ctx_managers:
            stack.enter_context(ctx)
        yield
