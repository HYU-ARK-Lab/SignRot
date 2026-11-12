# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import torch

import signrot.functional.kernel as kernel


class RMSNorm(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:  # fused pow/sum/rsqrt/mul, one launch
            return kernel.rmsnorm(x, self.eps, self.mean_dim)
        return self.forward_reference(x)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
