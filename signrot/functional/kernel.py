import torch
import signrot._CUDA

__all__ = [
    "matmul",  # int-4 matmul
    "sym_quant",
    "sym_dequant",
    "asym_quant",
    "asym_dequant",
    "asym_batch_dequant",
    "PackedQuantizedTensor",  # Quantization
]


class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel() // x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


class PackedQuantizedTensor:
    def __init__(
        self,
        quantized_x: torch.Tensor,
        scales_x: torch.Tensor,
        zeros_x: torch.Tensor | None = None,
    ):
        self.quantized_x = quantized_x
        self.scales_x = scales_x
        self.zeros_x = zeros_x

    def size(self):
        return self.quantized_x.size()

    @property
    def device(self):
        return self.quantized_x.device

    @property
    def dtype(self):
        return self.quantized_x.dtype


def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last


def flatten_last_two_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last_two, shape_last_two = x.shape[:-2], x.shape[-2:]
    x = x.view(-1, x.shape[-2], x.shape[-1])
    return x, shape_excl_last_two, shape_last_two


def matmul_si4_si4(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(
        A.shape[-1]
    )
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return signrot._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def matmul_ui4_si4(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(
        A.shape[-1]
    )
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return signrot._CUDA.matmul_ui4_si4(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def matmul_grouped(problem_sizes, ptr_a, ptr_b, ptr_c, lda, ldb, ldc, signed_a=True):
    """Grouped 4-bit GEMM: C_i = A_i @ B_i^T per problem in one launch.

    All arguments are device tensors: problem_sizes int32 [P, 3] rows (m, n, k),
    the rest int64 [P] device addresses / leading dims.  Results are written
    through ptr_c; per-element output is bit-identical to matmul_si4_si4 /
    matmul_ui4_si4 on each problem.  Nothing is read on the host, so the call
    is CUDA-graph capturable with data-dependent per-problem sizes.
    """
    signrot._CUDA.matmul_grouped(problem_sizes, ptr_a, ptr_b, ptr_c, lda, ldb, ldc, signed_a)


def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return signrot._CUDA.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)


def sym_quant_dynamic(x, clip_ratio=1.0):
    """Fused per-row amax + scale + symmetric int4 quantize/pack.

    Equivalent to the amax/div/cast/mul/sym_quant chain in nn.Quantizer, in one
    launch.  Returns (packed_q, scale) with scale shaped (*x.shape[:-1], 1).
    """
    assert x.dtype == torch.float16
    x, shape_excl_last = flatten_last_dim_and_return_shape(x.contiguous())
    q, scale = signrot._CUDA.sym_quant_dynamic(x, clip_ratio)
    return q.view(*shape_excl_last, -1), scale.view(*shape_excl_last, 1)


def asym_quant_dynamic(x, clip_ratio=1.0):
    """Fused per-row amin/amax + scale/zero + asymmetric uint4 quantize/pack."""
    assert x.dtype == torch.float16
    x, shape_excl_last = flatten_last_dim_and_return_shape(x.contiguous())
    q, scale, zero = signrot._CUDA.asym_quant_dynamic(x, clip_ratio)
    return (q.view(*shape_excl_last, -1),
            scale.view(*shape_excl_last, 1),
            zero.view(*shape_excl_last, 1))


def sym_quant_dynamic_moe(x, sign, expert_ids, clip_ratio=1.0):
    """sym_quant_dynamic over expert-sorted rows with a fused per-expert
    had_adj sign flip: row r is multiplied by sign[expert_ids[r]] (exact for
    +-1) before quantization.  sign may be None."""
    assert x.dtype == torch.float16
    return signrot._CUDA.sym_quant_dynamic_moe(x.contiguous(), sign, expert_ids, clip_ratio)


def asym_quant_dynamic_moe(x, sign, expert_ids, clip_ratio=1.0):
    """asym_quant_dynamic variant of sym_quant_dynamic_moe -> (q, scale, zero)."""
    assert x.dtype == torch.float16
    return signrot._CUDA.asym_quant_dynamic_moe(x.contiguous(), sign, expert_ids, clip_ratio)


def sym_dequant_moe(q, scale_row, w_scales, expert_ids, zero_row=None, w_sum=None):
    """Expert-indexed dequant with the asym zero correction fused in:

    out[r, c] = (q[r, c] - zero_row[r] * w_sum[e, c]) * scale_row[r] * w_scales[e, c]
    with e = expert_ids[r]; the zero/w_sum term is skipped when None.
    """
    assert q.dtype == torch.int32
    return signrot._CUDA.sym_dequant_moe(q, scale_row.contiguous(), w_scales,
                                         expert_ids, zero_row, w_sum)


def kv_quant_pack(x):
    """Fused KV-cache asymmetric uint4 quantize + pack.

    Returns (packed_q, param) where param[..., 0] is the scale and param[..., 1]
    the zero point -- the layout the flashinfer kv kernels expect.
    """
    assert x.dtype == torch.float16
    return signrot._CUDA.kv_quant_pack(x.contiguous())


def rope_qk(q, k, cos, sin):
    """Fused in-place rotary embedding for q [B,T,Hq,D] and k [B,T,Hk,D].

    Equivalent to HF's apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2),
    in one launch instead of ten elementwise ops.
    """
    signrot._CUDA.rope_qk(q, k, cos.contiguous(), sin.contiguous())
    return q, k


def swiglu(gate, up):
    """Fused silu(gate) * up."""
    return signrot._CUDA.swiglu(gate.contiguous(), up.contiguous())


def rmsnorm(x, eps, mean_dim):
    """Fused RMSNorm over the last dimension (fp32 accumulation, fp16 in/out)."""
    assert x.dtype == torch.float16
    return signrot._CUDA.rmsnorm(x.contiguous(), eps, mean_dim)


def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return signrot._CUDA.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(
        *q_shape_excl_last, -1
    )


def asym_quant_f16_si4(q, scale, zero):
    assert q.dtype == torch.float16
    assert scale.dtype == zero.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return signrot._CUDA.asym_quant_f16_si4(q, scale.view(-1), zero.view(-1)).view(
        *q_shape_excl_last, -1
    )


def asym_quant_f16_ui4(q, scale, zero):
    assert q.dtype == torch.float16
    assert scale.dtype == zero.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return signrot._CUDA.asym_quant_f16_ui4(q, scale.view(-1), zero.view(-1)).view(
        *q_shape_excl_last, -1
    )


def sum_i4(x, is_signed=True, dtype=torch.int32) -> torch.Tensor:
    assert x.dtype == torch.uint8, "packed int4 tensor must be torch.uint8"
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    out = signrot._CUDA.sum_i4(x, is_signed).view(*x_shape_excl_last, 1)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    return out


def matmul_colrow_i8_i32(a_col, b_row) -> torch.Tensor:
    assert a_col.dtype == torch.int8, "a_col must be torch.int8"
    assert b_row.dtype == torch.int32, "b_row must be torch.int32"

    a_col, a_shape_excl_last = flatten_last_dim_and_return_shape(a_col)
    assert a_col.shape[1] == 1, "a_col must have last dimension == 1"

    if b_row.dim() == 1:
        b_row = b_row.view(1, -1)
    assert b_row.dim() == 2 and b_row.shape[0] == 1, "b_row must have shape [1, C]"

    out = signrot._CUDA.matmul_colrow_i8_i32(a_col, b_row)
    return out.view(*a_shape_excl_last, b_row.shape[1])
