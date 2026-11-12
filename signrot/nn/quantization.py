from typing import TYPE_CHECKING

import torch
from signrot.functional.quantization import get_minq_maxq
import signrot.functional.kernel as kernel

class QuantizerBase(torch.nn.Module):
    def __init__(self, *, clip_ratio=1.0):
        super().__init__()
        self.clip_ratio = clip_ratio
    
    def forward(self, x):
        return x

class Quantizer(QuantizerBase):
    def __init__(self, *, clip_ratio=1.0):
        super().__init__(clip_ratio=clip_ratio)

    def forward(self, x) -> kernel.PackedQuantizedTensor:
        quantized_x, scales_x = kernel.sym_quant_dynamic(x, self.clip_ratio)
        return kernel.PackedQuantizedTensor(quantized_x, scales_x)

    def forward_reference(self, x) -> kernel.PackedQuantizedTensor:
        """Unfused version, kept as the equivalence reference for the kernel."""
        max_val = torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)
        scales_x = (max_val / 7).to(torch.float16) * self.clip_ratio
        quantized_x = kernel.sym_quant(x, scales_x)
        return kernel.PackedQuantizedTensor(quantized_x, scales_x)


class SymQuantizer(QuantizerBase):
    def __init__(self, *, bits: int = 4, clip_ratio=1.0):
        super().__init__(clip_ratio=clip_ratio)
        self.maxq = get_minq_maxq(bits, sym=True)[1]

    def forward(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor:
        # the fused kernel packs to int4 and applies clip_ratio after the
        # division (Quantizer's order); fall back when either differs
        if int(self.maxq) == 7 and self.clip_ratio == 1.0:
            quantized_x, scales_x = kernel.sym_quant_dynamic(x, self.clip_ratio)
            return kernel.PackedQuantizedTensor(quantized_x, scales_x)
        return self.forward_reference(x)

    def forward_reference(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor:
        max_val = x.abs().amax(dim=-1).unsqueeze(1) * self.clip_ratio
        scales_x = (max_val / self.maxq).to(torch.float16)
        quantized_x = kernel.sym_quant(x, scales_x)
        return kernel.PackedQuantizedTensor(quantized_x, scales_x)

    def extra_repr(self) -> str:
        return f"maxq={self.maxq}, clip_ratio={self.clip_ratio}"

    if TYPE_CHECKING:

        def __call__(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor: ...


class AsymQuantizer(QuantizerBase):
    def __init__(self, *, bits: int = 4, clip_ratio: float = 1.0):
        super().__init__(clip_ratio=clip_ratio)
        self.maxq = get_minq_maxq(bits, sym=False)[1]

    def forward(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor:
        quantized_x, scales_x, zeros_x = kernel.asym_quant_dynamic(x, self.clip_ratio)
        return kernel.PackedQuantizedTensor(quantized_x, scales_x, zeros_x.to(torch.int8))

    def forward_reference(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor:
        """Unfused version, kept as the equivalence reference for the kernel."""
        min_val = x.amin(dim=-1, keepdim=True) * self.clip_ratio
        max_val = x.amax(dim=-1, keepdim=True) * self.clip_ratio

        scales_x = (max_val - min_val) / self.maxq
        zeros_x = (-min_val / scales_x).round()

        quantized_x = kernel.asym_quant_f16_ui4(x, scales_x, zeros_x)
        return kernel.PackedQuantizedTensor(quantized_x, scales_x, zeros_x.to(torch.int8))

    def extra_repr(self) -> str:
        return f"maxq={self.maxq}, clip_ratio={self.clip_ratio}"

    if TYPE_CHECKING:

        def __call__(self, x: torch.Tensor) -> kernel.PackedQuantizedTensor: ...
