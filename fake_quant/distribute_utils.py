import math
import logging
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any

from accelerate import Accelerator, PartialState, dispatch_model, infer_auto_device_map
from accelerate.utils import (
    get_max_memory,
    get_balanced_memory,
    broadcast,
    gather_object,
    wait_for_everyone,
)
import torch
import torch.distributed as dist
import tqdm

import utils


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model._no_split_modules,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model._no_split_modules
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    utils.cleanup_memory()


def dynamic_distribute_model(
    model, accelerator: Accelerator, process_max_memory=None, device_map=None
) -> dict:
    """Dynamically distribute and parallelize the model across available GPUs based on their free memory.
    and create sub process groups for each model copy.

    Returns:
        dict: main processor rank : assigned gpu(or processor) list
    """
    local_rank = accelerator.process_index
    num_processes = accelerator.num_processes

    if process_max_memory is None:
        process_max_memory_wrap = []
        if accelerator.is_main_process:
            print("Dynamically distributing the model...")
            process_max_memory_wrap = [calculate_gpus_per_model(model, num_processes)]
        accelerator.wait_for_everyone()
        process_max_memory = gather_object(process_max_memory_wrap)[0]

    if local_rank in process_max_memory:  # group main processor
        max_memory = process_max_memory[local_rank]
        no_split_module_classes = model._no_split_modules
        if device_map is None:
            device_map = infer_auto_device_map(
                model, max_memory, no_split_module_classes
            )
        dispatch_model(model, device_map=device_map)
        utils.cleanup_memory()

    accelerator.wait_for_everyone()

    return {
        group_main_rank: list(group.keys())
        for group_main_rank, group in process_max_memory.items()
    }


@contextmanager
def split_with_rank[T: list | tuple | dict | torch.Tensor](
    inputs: T, selected_ranks=None, apply_padding=False  # type: ignore
):
    """Make selected ranks in the accelerator to form a new group and synchronize."""
    state = PartialState()
    if state.num_processes == 1:
        yield {state.process_index: inputs}
        return

    if selected_ranks is None:
        selected_ranks = range(state.num_processes)
    selected_ranks = list(set(selected_ranks))

    length = len(inputs)
    num_selected = len(selected_ranks)
    num_samples_per_process = math.ceil(length / num_selected)

    def _split_values(inputs, start_index, end_index) -> T:
        if isinstance(inputs, (list, tuple, torch.Tensor)):
            result = inputs[start_index:end_index]
            if apply_padding:
                if isinstance(result, torch.Tensor):
                    from accelerate.utils import pad_across_processes, send_to_device

                    # The tensor needs to be on the device before we can pad it
                    tensorized_result = send_to_device(result, state.device)
                    result = pad_across_processes(
                        tensorized_result, pad_index=inputs[-1]  # type: ignore
                    )
                else:
                    # FIXME: apply padding on non_selected ranks
                    result += [result[-1]] * (num_samples_per_process - len(result))  # type: ignore
            return result  # type: ignore
        elif isinstance(inputs, dict):
            for key in inputs.keys():
                inputs[key] = _split_values(inputs[key], start_index, end_index)
            return inputs  # type: ignore
        else:
            return inputs

    rank_map = {}
    for rank in range(state.num_processes):
        rank_order = (
            selected_ranks.index(rank) if rank in selected_ranks else num_selected
        )
        start_index = rank_order * num_samples_per_process
        end_index = start_index + num_samples_per_process
        rank_map[rank] = _split_values(inputs, start_index, end_index)

    yield rank_map


@contextmanager
def split_between_selected[T: list | tuple | dict | torch.Tensor](
    inputs: T, selected_ranks=None, apply_padding=False  # type: ignore
):
    """Make selected ranks in the accelerator to form a new group and synchronize."""
    state = PartialState()
    local_rank = state.process_index
    with split_with_rank(inputs, selected_ranks, apply_padding) as rank_map:
        yield rank_map[local_rank]


def sync_state_dict[T: dict](
    state_dict: T,
    device=None,
    src=0,
    *,
    tqdm_kwargs={
        "desc": "Syncing tensors",
    },
) -> T:
    desc = tqdm_kwargs.get("desc", "Syncing tensors")
    for key in (pbar := tqdm.tqdm(state_dict.keys(), **tqdm_kwargs)):
        pbar.set_description(desc + f" ({key})")
        state_dict[key] = broadcast(
            state_dict[key].contiguous().to(device), from_process=src
        )
    return state_dict


def sync_module[T: torch.nn.Module](
    module: T, device=None, src=0, tqdm_kwargs={"desc": "Syncing module"}
) -> T:
    module.load_state_dict(
        sync_state_dict(module.state_dict(), device, src, tqdm_kwargs=tqdm_kwargs)
    )
    return module


def calculate_gpus_per_model(model, num_processes: int, gpu_mem_scale=0.9) -> dict:
    # consider KV cache or intermediate activations
    required_mem_bytes = model.get_memory_footprint()
    num_gpus = torch.cuda.device_count()

    # to avoid OOM, we consider only max memory * gpu_mem_scale
    gpu_mem_bytes = [
        (gpu_index, torch.cuda.mem_get_info(gpu_index)) for gpu_index in range(num_gpus)
    ]

    # dynamically group processes based on available GPU memory
    processor_max_memory = {}
    for group_main_rank in range(num_processes):
        if len(gpu_mem_bytes) == 0:
            break

        # main gpu for the group is not shared with other groups
        group_main_gpu, (free_mem, total_mem) = gpu_mem_bytes.pop(0)
        free_available_mem = int(free_mem - total_mem * (1 - gpu_mem_scale))
        max_memory = {group_main_gpu: free_available_mem}

        while sum(max_memory.values()) < required_mem_bytes and len(gpu_mem_bytes) > 0:
            remaining_required = required_mem_bytes - sum(max_memory.values())
            gpu_index, (free_mem, total_mem) = gpu_mem_bytes[-1]
            free_available_mem = int(free_mem - total_mem * (1 - gpu_mem_scale))
            mem_allocated = min(free_available_mem, remaining_required)
            # add required memory
            max_memory[gpu_index] = mem_allocated

            if free_available_mem - mem_allocated > 0:
                gpu_mem_bytes[-1] = (gpu_index, (free_mem - mem_allocated, total_mem))
            else:
                gpu_mem_bytes.pop()

        if sum(max_memory.values()) >= required_mem_bytes:
            balanced_max_memory = get_balanced_memory(
                model, max_memory, model._no_split_modules  # type: ignore
            )
            processor_max_memory[group_main_rank] = balanced_max_memory

    return processor_max_memory


def wait_every_group_member(group=None) -> None:
    if group is None:
        wait_for_everyone()
    else:
        dist.barrier(group=group)


def get_process_group_ranks(group=None) -> list:
    if group is None:
        return list(range(PartialState().num_processes))
    return dist.get_process_group_ranks(group)


def group_gather_object(obj: Any, group=None) -> list:
    if group is None:
        return gather_object(obj)

    output_objects = [None for _ in get_process_group_ranks(group)]
    torch.distributed.all_gather_object(output_objects, obj, group)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]  # type: ignore


def group_gpu(model, max_num_groups: int, strategy: str = "greedy") -> dict:
    """Group GPUs for each model copy based on their memory capacity.

    Args:
        model: The model to be distributed.
        max_num_groups (int): Number of model copies/groups.
        strategy (str): Strategy for grouping GPUs.
        - 'balanced': Each group gets an equal share of each GPU's memory.
        - 'greedy': Each group gets the maximum available memory from the GPUs in a round-robin fashion.
    Returns:
        dict: main processor rank : assigned gpu(or processor) list
    """
    max_memory = get_max_memory()
    group_max_memory = OrderedDict()
    del max_memory["cpu"]
    max_memory_list = sorted(max_memory.keys(), key=lambda x: x)
    model_required_mem = model.get_memory_footprint()

    if strategy == "balanced":
        total_memory = sum(max_memory.values())
        possible_max_groups = min(total_memory // model_required_mem, max_num_groups)
        for group_id in range(possible_max_groups):
            group_max_memory[group_id] = OrderedDict()
            for gpu_index in max_memory_list:
                group_max_memory[group_id][gpu_index] = max_memory[gpu_index] // possible_max_groups

            group_max_memory[group_id] = get_balanced_memory(
                model,
                group_max_memory[group_id],
                no_split_module_classes=model._no_split_modules,
            )

    elif strategy == "greedy":
        for group_id in range(max_num_groups):
            mm = OrderedDict()
            mm[group_id] = max_memory[group_id]
            max_memory_list.remove(group_id)

            while sum(mm.values()) < model_required_mem:
                gpu_index = max_memory_list.pop()
                mm[gpu_index] = max_memory[gpu_index]
            if sum(mm.values()) >= model_required_mem:
                group_max_memory[group_id] = mm

        for group_rank, max_memory in group_max_memory.items():
            # balance memory within the group
            balanced_mem = get_balanced_memory(
                model, max_memory, no_split_module_classes=model._no_split_modules
            )
            group_max_memory[group_rank] = balanced_mem

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Must be 'balanced' or 'greedy'."
        )

    return group_max_memory


def ddp_model(model, accelerator: Accelerator, load_model):
    local_rank = accelerator.process_index
    process_max_memory = {}
    model_mem_require = model.get_memory_footprint()
    num_gpus = torch.cuda.device_count()
    gpu_buffer_bytes = OrderedDict(
        [
            (gpu_index, int(torch.cuda.mem_get_info(gpu_index)[1] * 0.1))
            for gpu_index in range(num_gpus)
        ]
    )
    gpu_buffer_bytes["cpu"] = 0

    for process_rank in range(accelerator.num_processes):
        max_memory = {k: v - gpu_buffer_bytes[k] for k, v in get_max_memory().items()}
        if process_rank == local_rank:
            print(
                f"[rank {local_rank}] DDP loading model for process {process_rank}..."
            )
            print(f"[rank {local_rank}] DDP current max memory: {max_memory}")
            # reorder to accelerator.device is on top
            max_memory_ordered = OrderedDict(
                sorted(
                    max_memory.items(),
                    key=lambda x: (
                        0
                        if x == accelerator.device.index
                        else (
                            1 + accelerator.device.index
                            if accelerator.device.index is not None
                            else 1
                        )
                    ),
                )
            )
            print(f"[rank {local_rank}] DDP reordered max memory: {max_memory_ordered}")
            # load model
            device_map = infer_auto_device_map(model, max_memory=max_memory_ordered)
            # if cpu needs, cut loading
            if "cpu" in device_map.values():
                print(
                    f"[rank {local_rank}] DDP detected cpu offload in device_map, skipping loading for process {process_rank}...",
                    device_map,
                )
                accelerator.wait_for_everyone()
                break
            print(f"[rank {local_rank}] DDP device_map: {device_map}")
            load_model(model, device_map)
        accelerator.wait_for_everyone()

        max_memory_after = get_max_memory()
        diff = {
            k: max_memory[k] - max_memory_after[k] - gpu_buffer_bytes[k]
            for k in max_memory.keys()
            if k != "cpu"
        }
        process_max_memory[process_rank] = diff

    return process_max_memory
