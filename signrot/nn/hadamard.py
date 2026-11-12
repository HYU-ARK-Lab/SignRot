# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
from signrot.functional.hadamard import matmul_hadU_cuda, get_hadK
from signrot.nn.linear import Linear4bit
import torch


class OnlineHadamard(torch.nn.Module):
    had_adj_sign: torch.Tensor | None

    def __init__(self, hadamard_dim, tied_module: Linear4bit|None=None, force_fp32=False, had_adj=False, dynamic_adj=False):
        super().__init__()
        self.fp32_had = force_fp32
        had_rem_dim, self.rem_dim = get_hadK(hadamard_dim)
        if had_rem_dim is not None:
            self.register_buffer("had_rem_dim", had_rem_dim)
            if not self.fp32_had:
                self.had_rem_dim = self.had_rem_dim.to(torch.float16)
        else:
            self.had_rem_dim = None
        self.register_buffer("had_adj_sign", None)
        if had_adj:
            self.register_buffer("had_adj_sign", torch.ones(hadamard_dim, dtype=torch.int8))
        self.dynamic_adj = dynamic_adj
        self.module: Linear4bit|None = tied_module

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        if self.fp32_had:
            x = x.float()
        x = matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim)

        if self.had_adj_sign is not None:
            if not self._had_adj_ready_cached: # or self.dynamic_adj:  # not initialized
                raise RuntimeError("Dynamic hadamard adjustment not implemented yet.")
            #     single_batch = x[0] if x.dim() == 3 else x
            #     ma_channel_idx = single_batch.abs().max(dim=-1).values.argmax()
            #     self.register_buffer("had_adj_sign", ((single_batch[ma_channel_idx] >= 0) * 2 - 1).char())
            #     self.module.weight.data = self.module.weight * self.had_adj_sign
            x = x * self.had_adj_sign

        x = x.to(x_dtype)
        return x

    @property
    def _had_adj_ready_cached(self):
        # `had_adj_ready` reads a device tensor into a Python bool, i.e. a
        # device->host sync on every down_proj call.  The sign vector is frozen
        # once the kernel modules are built, so the check is done once.
        ready = getattr(self, "_had_adj_ready_flag", None)
        if ready is None:
            ready = self.had_adj_ready()
            self._had_adj_ready_flag = ready
        return ready

    def had_adj_ready(self):
        return bool(self.had_adj_sign is not None and torch.all(self.had_adj_sign != 0))
