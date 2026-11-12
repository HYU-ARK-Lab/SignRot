// Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <gemm_grouped.h>
#include <quant.h>
#include <fused.h>
#include <moe.h>
#include <flashinfer.h>


torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B)
{
    torch::checkAllContiguous("matmul", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("matmul", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    matmul_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

    return C;
}

torch::Tensor matmul_ui4_si4(const torch::Tensor &A, const torch::Tensor &B)
{
    torch::checkAllContiguous("matmul_ui4_si4", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("matmul_ui4_si4", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("matmul_ui4_si4", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * kElementsPerVectorUInt4;  // 4bit packing is on the columns
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    matmul_host_ui4_si4(A.data_ptr<UInt4Storage>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

    return C;
}

void matmul_grouped(const torch::Tensor &problem_sizes,
                    const torch::Tensor &ptr_A, const torch::Tensor &ptr_B,
                    const torch::Tensor &ptr_C, const torch::Tensor &lda,
                    const torch::Tensor &ldb, const torch::Tensor &ldc,
                    bool signed_a)
{
    torch::checkAllContiguous("matmul_grouped", {{problem_sizes, "problem_sizes", 0},
                                                 {ptr_A, "ptr_A", 1},
                                                 {ptr_B, "ptr_B", 2},
                                                 {ptr_C, "ptr_C", 3},
                                                 {lda, "lda", 4},
                                                 {ldb, "ldb", 5},
                                                 {ldc, "ldc", 6}});
    torch::checkDeviceType("matmul_grouped",
                           {problem_sizes, ptr_A, ptr_B, ptr_C, lda, ldb, ldc},
                           at::DeviceType::CUDA);
    TORCH_CHECK(problem_sizes.dtype() == torch::kInt32 && problem_sizes.dim() == 2 &&
                problem_sizes.size(1) == 3,
                "problem_sizes must be int32 [problem_count, 3] (m, n, k) rows");
    int problem_count = problem_sizes.size(0);
    for (const auto *t : {&ptr_A, &ptr_B, &ptr_C, &lda, &ldb, &ldc})
    {
        TORCH_CHECK((*t).dtype() == torch::kInt64 && (*t).numel() == problem_count,
                    "pointer/ld arrays must be int64 with one entry per problem");
    }

    auto *sizes = reinterpret_cast<const cutlass::gemm::GemmCoord *>(
            problem_sizes.data_ptr<int32_t>());
    if (signed_a)
    {
        matmul_grouped_si4_si4_host(sizes, problem_count,
                                    ptr_A.data_ptr<int64_t>(), ptr_B.data_ptr<int64_t>(),
                                    ptr_C.data_ptr<int64_t>(), lda.data_ptr<int64_t>(),
                                    ldb.data_ptr<int64_t>(), ldc.data_ptr<int64_t>());
    }
    else
    {
        matmul_grouped_ui4_si4_host(sizes, problem_count,
                                    ptr_A.data_ptr<int64_t>(), ptr_B.data_ptr<int64_t>(),
                                    ptr_C.data_ptr<int64_t>(), lda.data_ptr<int64_t>(),
                                    ldb.data_ptr<int64_t>(), ldc.data_ptr<int64_t>());
    }
}

static const int8_t *moe_sign_ptr(const std::optional<torch::Tensor> &sign,
                                  uint32_t colsSrc, const char *name)
{
    if (!sign.has_value())
    {
        return nullptr;
    }
    torch::checkAllContiguous(name, {{*sign, "sign", 10}});
    torch::checkDeviceType(name, {*sign}, at::DeviceType::CUDA);
    TORCH_CHECK(sign->dtype() == torch::kInt8 && sign->dim() == 2 &&
                sign->size(1) == colsSrc,
                name, ": sign must be int8 [num_experts, cols]");
    return sign->data_ptr<int8_t>();
}

std::tuple<torch::Tensor, torch::Tensor> sym_quant_dynamic_moe(
    const torch::Tensor &x, const std::optional<torch::Tensor> &sign,
    const torch::Tensor &expert_ids, const double clip_ratio)
{
    torch::checkAllContiguous("sym_quant_dynamic_moe", {{x, "x", 0}, {expert_ids, "expert_ids", 1}});
    torch::checkDeviceType("sym_quant_dynamic_moe", {x, expert_ids}, at::DeviceType::CUDA);
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf,
                "sym_quant_dynamic_moe expects a 2D fp16 [rows, cols] tensor");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt32 && expert_ids.numel() == x.size(0),
                "expert_ids must be int32 with one entry per row");

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));

    sym_quant_dynamic_moe_host((half *)x.data_ptr(),
                               moe_sign_ptr(sign, colsSrc, "sym_quant_dynamic_moe"),
                               expert_ids.data_ptr<int32_t>(), rows, colsSrc, colsDst,
                               (float)clip_ratio, (half *)scale.data_ptr(),
                               q.data_ptr<Int4Storage>());
    return {q, scale};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asym_quant_dynamic_moe(
    const torch::Tensor &x, const std::optional<torch::Tensor> &sign,
    const torch::Tensor &expert_ids, const double clip_ratio)
{
    torch::checkAllContiguous("asym_quant_dynamic_moe", {{x, "x", 0}, {expert_ids, "expert_ids", 1}});
    torch::checkDeviceType("asym_quant_dynamic_moe", {x, expert_ids}, at::DeviceType::CUDA);
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf,
                "asym_quant_dynamic_moe expects a 2D fp16 [rows, cols] tensor");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt32 && expert_ids.numel() == x.size(0),
                "expert_ids must be int32 with one entry per row");

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVectorUInt4);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));
    auto zero = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));

    asym_quant_dynamic_moe_host((half *)x.data_ptr(),
                                moe_sign_ptr(sign, colsSrc, "asym_quant_dynamic_moe"),
                                expert_ids.data_ptr<int32_t>(), rows, colsSrc, colsDst,
                                (float)clip_ratio, (half *)scale.data_ptr(),
                                (half *)zero.data_ptr(), q.data_ptr<UInt4Storage>());
    return {q, scale, zero};
}

torch::Tensor sym_dequant_moe(
    const torch::Tensor &q, const torch::Tensor &scale_row,
    const torch::Tensor &w_scales, const torch::Tensor &expert_ids,
    const std::optional<torch::Tensor> &zero_row,
    const std::optional<torch::Tensor> &w_sum)
{
    torch::checkAllContiguous("sym_dequant_moe", {{q, "q", 0},
                                                  {scale_row, "scale_row", 1},
                                                  {w_scales, "w_scales", 2},
                                                  {expert_ids, "expert_ids", 3}});
    torch::checkDeviceType("sym_dequant_moe", {q, scale_row, w_scales, expert_ids},
                           at::DeviceType::CUDA);
    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);
    TORCH_CHECK(q.dim() == 2 && q.dtype() == torch::kInt32, "q must be int32 [rows, cols]");
    TORCH_CHECK(scale_row.dtype() == torch::kHalf && scale_row.numel() == rows,
                "scale_row must be fp16 with one entry per row");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf && w_scales.dim() == 2 &&
                w_scales.size(1) == cols,
                "w_scales must be fp16 [num_experts, cols]");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt32 && expert_ids.numel() == rows,
                "expert_ids must be int32 with one entry per row");
    TORCH_CHECK(zero_row.has_value() == w_sum.has_value(),
                "zero_row and w_sum must be given together");

    const int8_t *zero_ptr = nullptr;
    const int32_t *wsum_ptr = nullptr;
    if (zero_row.has_value())
    {
        torch::checkAllContiguous("sym_dequant_moe", {{*zero_row, "zero_row", 4},
                                                      {*w_sum, "w_sum", 5}});
        TORCH_CHECK(zero_row->dtype() == torch::kInt8 && zero_row->numel() == rows,
                    "zero_row must be int8 with one entry per row");
        TORCH_CHECK(w_sum->dtype() == torch::kInt32 && w_sum->dim() == 2 &&
                    w_sum->size(1) == cols && w_sum->size(0) == w_scales.size(0),
                    "w_sum must be int32 [num_experts, cols]");
        zero_ptr = zero_row->data_ptr<int8_t>();
        wsum_ptr = w_sum->data_ptr<int32_t>();
    }

    auto out = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));
    sym_dequant_moe_host(q.data_ptr<int32_t>(), (half *)scale_row.data_ptr(),
                         (half *)w_scales.data_ptr(), expert_ids.data_ptr<int32_t>(),
                         zero_ptr, wsum_ptr, rows, cols, (half *)out.data_ptr());
    return out;
}

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale)
{
    torch::checkAllContiguous("sym_quant", {{x,     "x",     0},
                                                      {scale, "scale", 1}});
    torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

    torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
    torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst},torch::dtype(torch::kUInt8).device(x.device()));

    sym_quant_host((half*)x.data_ptr(), (half*)scale.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<Int4Storage>());

    return q;
}


torch::Tensor sym_dequant(const torch::Tensor &q,
                                     const torch::Tensor &scale_row,
                                     const torch::Tensor &scale_col,
                                     const int bits)
{
    torch::checkAllContiguous("sym_dequant",
                              {{q,         "q",         0},
                               {scale_row, "scale_row", 1},
                               {scale_col, "scale_col", 2}
                              });
    torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                           at::DeviceType::CUDA);

    torch::checkAllSameGPU("sym_dequant",
                           {{q,         "q",         0},
                            {scale_row, "scale_row", 1},
                            {scale_col, "scale_col", 2}
                           });

    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);

    torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                     rows);
    torch::checkSize("sym_dequant", torch::TensorArg{scale_col, "scale_col", 2}, 0,
                     cols);

    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            sym_dequant_host(q.data_ptr<int32_t>(), (half*)scale_row.data_ptr(), (half*)scale_col.data_ptr(),
                    rows, cols, (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

    return x;
}

torch::Tensor asym_quant_f16_si4(
    const torch::Tensor &x,
    const torch::Tensor &scale, const torch::Tensor &zeros
)
{
    torch::checkAllContiguous("asym_quant_f16_si4", {{x, "x", 0}, {scale, "scale", 1}, {zeros, "zeros", 2}});
    torch::checkDeviceType("asym_quant_f16_si4", {x, scale, zeros}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("asym_quant_f16_si4", {{x, "x", 0}, {scale, "scale", 1} , {zeros, "zeros", 2}});
    torch::checkSize("asym_quant_f16_si4", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
    torch::checkSize("asym_quant_f16_si4", torch::TensorArg{zeros, "zeros", 2}, 0, x.size(0));

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));

    asym_quant_f16_si4_host((half*)x.data_ptr(), (half*)scale.data_ptr(), (half*)zeros.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<Int4Storage>());
    
    return q;
}

torch::Tensor asym_quant_f16_ui4(
    const torch::Tensor &x,
    const torch::Tensor &scale, const torch::Tensor &zeros
)
{
    torch::checkAllContiguous("asym_quant_f16_ui4", {{x, "x", 0}, {scale, "scale", 1}, {zeros, "zeros", 2}});
    torch::checkDeviceType("asym_quant_f16_ui4", {x, scale, zeros}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("asym_quant_f16_ui4", {{x, "x", 0}, {scale, "scale", 1} , {zeros, "zeros", 2}});
    torch::checkSize("asym_quant_f16_ui4", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
    torch::checkSize("asym_quant_f16_ui4", torch::TensorArg{zeros, "zeros", 2}, 0, x.size(0));

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVectorUInt4);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));

    asym_quant_f16_ui4_host((half*)x.data_ptr(), (half*)scale.data_ptr(), (half*)zeros.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<UInt4Storage>());
    
    return q;
}

// ===== Fused pre-GEMM kernels =====

std::tuple<torch::Tensor, torch::Tensor> sym_quant_dynamic(
    const torch::Tensor &x, const double clip_ratio)
{
    torch::checkAllContiguous("sym_quant_dynamic", {{x, "x", 0}});
    torch::checkDeviceType("sym_quant_dynamic", {x}, at::DeviceType::CUDA);
    TORCH_CHECK(x.dim() == 2, "sym_quant_dynamic expects a 2D [rows, cols] tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "sym_quant_dynamic expects fp16 input");

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));

    sym_quant_dynamic_host((half *)x.data_ptr(), rows, colsSrc, colsDst,
                           (float)clip_ratio, (half *)scale.data_ptr(),
                           q.data_ptr<Int4Storage>());
    return {q, scale};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> asym_quant_dynamic(
    const torch::Tensor &x, const double clip_ratio)
{
    torch::checkAllContiguous("asym_quant_dynamic", {{x, "x", 0}});
    torch::checkDeviceType("asym_quant_dynamic", {x}, at::DeviceType::CUDA);
    TORCH_CHECK(x.dim() == 2, "asym_quant_dynamic expects a 2D [rows, cols] tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "asym_quant_dynamic expects fp16 input");

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVectorUInt4);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));
    auto zero = torch::empty({rows, 1}, torch::dtype(torch::kHalf).device(x.device()));

    asym_quant_dynamic_host((half *)x.data_ptr(), rows, colsSrc, colsDst,
                            (float)clip_ratio, (half *)scale.data_ptr(),
                            (half *)zero.data_ptr(), q.data_ptr<UInt4Storage>());
    return {q, scale, zero};
}

std::tuple<torch::Tensor, torch::Tensor> kv_quant_pack(const torch::Tensor &x)
{
    torch::checkAllContiguous("kv_quant_pack", {{x, "x", 0}});
    torch::checkDeviceType("kv_quant_pack", {x}, at::DeviceType::CUDA);
    TORCH_CHECK(x.scalar_type() == at::kHalf, "kv_quant_pack expects fp16 input");

    uint32_t colsSrc = x.size(-1);
    TORCH_CHECK(colsSrc % 2 == 0, "kv_quant_pack expects an even last dimension");
    uint32_t colsDst = colsSrc / 2;
    uint32_t rows = x.numel() / colsSrc;

    auto q_shape = x.sizes().vec();
    q_shape.back() = colsDst;
    auto p_shape = x.sizes().vec();
    p_shape.back() = 2;

    auto q = torch::empty(q_shape, torch::dtype(torch::kUInt8).device(x.device()));
    auto param = torch::empty(p_shape, torch::dtype(torch::kHalf).device(x.device()));

    kv_quant_pack_host((half *)x.data_ptr(), rows, colsSrc, colsDst,
                       (half *)param.data_ptr(), q.data_ptr<uint8_t>());
    return {q, param};
}

void rope_qk(torch::Tensor q, torch::Tensor k,
             const torch::Tensor &cos, const torch::Tensor &sin)
{
    torch::checkAllContiguous("rope_qk", {{q, "q", 0}, {k, "k", 1}, {cos, "cos", 2}, {sin, "sin", 3}});
    torch::checkDeviceType("rope_qk", {q, k, cos, sin}, at::DeviceType::CUDA);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "rope_qk expects [B, T, H, D] q/k");
    TORCH_CHECK(cos.dim() == 3, "rope_qk expects [B or 1, T, D] cos/sin");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf,
                "rope_qk expects fp16 q/k");
    TORCH_CHECK(cos.scalar_type() == at::kHalf, "rope_qk expects fp16 cos/sin");

    uint32_t n_seqs = q.size(0), seq_len = q.size(1);
    uint32_t head_dim = q.size(3);
    TORCH_CHECK(k.size(0) == q.size(0) && k.size(1) == q.size(1) && k.size(3) == q.size(3),
                "q and k must agree on batch, sequence and head_dim");
    TORCH_CHECK(cos.size(1) == q.size(1) && cos.size(2) == q.size(3),
                "cos/sin must be [B or 1, T, head_dim]");
    TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == q.size(0),
                "cos/sin batch must be 1 or match q");
    TORCH_CHECK(head_dim % 2 == 0, "rope_qk expects an even head_dim");

    rope_qk_host((half *)q.data_ptr(), (half *)k.data_ptr(),
                 (half *)cos.data_ptr(), (half *)sin.data_ptr(),
                 n_seqs, seq_len, (uint32_t)cos.size(0),
                 (uint32_t)q.size(2), (uint32_t)k.size(2), head_dim);
}

torch::Tensor swiglu(const torch::Tensor &gate, const torch::Tensor &up)
{
    torch::checkAllContiguous("swiglu", {{gate, "gate", 0}, {up, "up", 1}});
    torch::checkDeviceType("swiglu", {gate, up}, at::DeviceType::CUDA);
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    TORCH_CHECK(gate.scalar_type() == at::kHalf, "swiglu expects fp16 input");
    auto out = torch::empty_like(gate);
    swiglu_host((half *)gate.data_ptr(), (half *)up.data_ptr(),
                (uint32_t)gate.numel(), (half *)out.data_ptr());
    return out;
}

torch::Tensor rmsnorm(const torch::Tensor &x, const double eps, const int64_t mean_dim)
{
    torch::checkAllContiguous("rmsnorm", {{x, "x", 0}});
    torch::checkDeviceType("rmsnorm", {x}, at::DeviceType::CUDA);
    TORCH_CHECK(x.scalar_type() == at::kHalf, "rmsnorm expects fp16 input");

    uint32_t cols = x.size(-1);
    uint32_t rows = x.numel() / cols;
    auto out = torch::empty_like(x);
    rmsnorm_host((half *)x.data_ptr(), rows, cols, (float)eps, (uint32_t)mean_dim,
                 (half *)out.data_ptr());
    return out;
}

torch::Tensor sum_i4(const torch::Tensor &x, const bool is_signed)
{
    torch::checkAllContiguous("sum_i4", {{x, "x", 0}});
    torch::checkDeviceType("sum_i4", {x}, at::DeviceType::CUDA);

    TORCH_CHECK(x.dim() == 2, "sum_i4 expects a 2D tensor of shape [rows, packed_cols]");
    TORCH_CHECK(x.scalar_type() == at::kByte, "sum_i4 expects torch.uint8 packed input");

    uint32_t rows = x.size(0);
    uint32_t packed_cols = x.size(1);

    auto out = torch::empty({rows, 1}, torch::dtype(torch::kInt32).device(x.device()));
    sum_i4_host(x.data_ptr<uint8_t>(), rows, packed_cols, is_signed, out.data_ptr<int32_t>());
    return out;
}

torch::Tensor matmul_colrow_i8_i32(const torch::Tensor &a_col, const torch::Tensor &b_row)
{
    torch::checkAllContiguous("matmul_colrow_i8_i32", {{a_col, "a_col", 0}, {b_row, "b_row", 1}});
    torch::checkDeviceType("matmul_colrow_i8_i32", {a_col, b_row}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_colrow_i8_i32", {{a_col, "a_col", 0}, {b_row, "b_row", 1}});

    TORCH_CHECK(a_col.dim() == 2, "a_col must be 2D with shape [N, 1]");
    TORCH_CHECK(b_row.dim() == 2, "b_row must be 2D with shape [1, C]");
    TORCH_CHECK(a_col.size(1) == 1, "a_col second dimension must be 1");
    TORCH_CHECK(b_row.size(0) == 1, "b_row first dimension must be 1");
    TORCH_CHECK(a_col.scalar_type() == at::kChar, "a_col must be torch.int8");
    TORCH_CHECK(b_row.scalar_type() == at::kInt, "b_row must be torch.int32");

    uint32_t rows = a_col.size(0);
    uint32_t cols = b_row.size(1);
    auto out = torch::empty({rows, cols}, torch::dtype(torch::kInt32).device(a_col.device()));
    matmul_colrow_i8_i32_host(
        a_col.data_ptr<int8_t>(),
        b_row.data_ptr<int32_t>(),
        rows,
        cols,
        out.data_ptr<int32_t>()
    );
    return out;
}

// ===== Flash Infer ======
inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)


void batch_decode_i4(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx,
                     std::optional<torch::Tensor> tmp = std::nullopt,
                     int64_t num_chunks = 1) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  float *tmp_ptr = nullptr;
  if (tmp.has_value() && num_chunks > 1) {
    CHECK_INPUT(tmp.value());
    CHECK_EQ(tmp->scalar_type(), at::ScalarType::Float);
    // [batch, heads, chunks, head_dim + 4] (padded for aligned vector stores)
    CHECK_EQ(tmp->numel(), (int64_t)batch_size * num_heads * num_chunks * (head_dim + 4));
    tmp_ptr = tmp->data_ptr<float>();
  } else {
    num_chunks = 1;
  }

  FlashInferBatchDecodeKernel_i4<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size, tmp_ptr, (int)num_chunks);
}

void init_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void batch_decode_f16(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_f16<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}



//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul", &matmul,
          "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
          "UINT8, CUDA))\n"
          "output: torch.Tensor(M x N, INT32, CUDA)\n"
          "output = int4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"));
    m.def("matmul_ui4_si4", &matmul_ui4_si4,
          "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, INT8, CUDA))\n"
          "output: torch.Tensor(M x N, INT32, CUDA)\n"
          "output = uint4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"));

    m.def("matmul_grouped", &matmul_grouped,
          "Grouped int4 GEMM: C_i = A_i @ B_i^T per problem, one launch.\n"
          "problem_sizes: torch.Tensor(P x 3, INT32, CUDA) rows (m, n, k)\n"
          "ptr_A/ptr_B/ptr_C, lda/ldb/ldc: torch.Tensor(P, INT64, CUDA) device\n"
          "addresses / leading dims; A row-major packed 4-bit (signed if\n"
          "signed_a else unsigned), B column-major packed int4, C int32.\n"
          "All metadata stays on the device (CUDA-graph capturable).",
          py::arg("problem_sizes"), py::arg("ptr_A"), py::arg("ptr_B"),
          py::arg("ptr_C"), py::arg("lda"), py::arg("ldb"), py::arg("ldc"),
          py::arg("signed_a"));

    m.def("sym_quant_dynamic_moe", &sym_quant_dynamic_moe,
          "sym_quant_dynamic with an optional fused per-expert sign flip:\n"
          "x: (rows x cols, FP16), sign: (E x cols, INT8) or None,\n"
          "expert_ids: (rows, INT32) -> (packed int4, scale)",
          py::arg("x"), py::arg("sign"), py::arg("expert_ids"), py::arg("clip_ratio"));
    m.def("asym_quant_dynamic_moe", &asym_quant_dynamic_moe,
          "asym_quant_dynamic with an optional fused per-expert sign flip:\n"
          "-> (packed uint4, scale, zero)",
          py::arg("x"), py::arg("sign"), py::arg("expert_ids"), py::arg("clip_ratio"));
    m.def("sym_dequant_moe", &sym_dequant_moe,
          "Expert-indexed dequant with fused asym zero correction:\n"
          "out[r, c] = (q[r, c] - zero_row[r] * w_sum[e, c]) * scale_row[r] *\n"
          "w_scales[e, c] with e = expert_ids[r]; zero_row/w_sum may be None",
          py::arg("q"), py::arg("scale_row"), py::arg("w_scales"),
          py::arg("expert_ids"), py::arg("zero_row"), py::arg("w_sum"));

    m.def("sym_quant", &sym_quant,
          "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
          "torch.Tensor(M x 1, FP16, CUDA))"
          "bits: int\n"
          "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
          "output = int4Packing(int4Rounding(source / scale)\n",
          py::arg("x"), py::arg("scale"));

    m.def("sym_dequant", &sym_dequant,
          "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
          "FP16), scale_col: torch.Tensor(1 x N, FP16)"
          "bits: int\n"
          "output: torch.Tensor(M x N, FP16)\n"
          "output = x * scale_row * scale_col"
          "when bits equal 8: "
          "input x type is int8\n"
          "when bits equal 16: "
          "input x type is FP16\n"
          "when bits equal 32: "
          "input x type is int32\n",
          py::arg("q"), py::arg("scale_row"), py::arg("scale_col"),
          py::arg("bits"));
    m.def("asym_quant_f16_si4", &asym_quant_f16_si4,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA), zeros: torch.Tensor(M x 1, FP16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"), py::arg("zeros"));
    m.def("asym_quant_f16_ui4", &asym_quant_f16_ui4,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA), zeros: torch.Tensor(M x 1, FP16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"), py::arg("zeros"));
    m.def("sym_quant_dynamic", &sym_quant_dynamic,
        "input: (x: torch.Tensor(M x N, FP16, CUDA), clip_ratio: float)\n"
        "output: (q: torch.Tensor(M x N/2, UINT8), scale: torch.Tensor(M x 1, FP16))\n"
        "Fused per-row amax + scale + symmetric int4 quantize/pack.",
        py::arg("x"), py::arg("clip_ratio") = 1.0);
    m.def("asym_quant_dynamic", &asym_quant_dynamic,
        "input: (x: torch.Tensor(M x N, FP16, CUDA), clip_ratio: float)\n"
        "output: (q: UINT8 M x N/2, scale: FP16 M x 1, zero: FP16 M x 1)\n"
        "Fused per-row amin/amax + scale/zero + asymmetric uint4 quantize/pack.",
        py::arg("x"), py::arg("clip_ratio") = 1.0);
    m.def("kv_quant_pack", &kv_quant_pack,
        "input: (x: torch.Tensor(..., D, FP16, CUDA))\n"
        "output: (q: UINT8 (..., D/2), param: FP16 (..., 2) = [scale, zero])\n"
        "Fused KV-cache asymmetric uint4 quantize + pack.",
        py::arg("x"));
    m.def("rope_qk", &rope_qk,
        "in-place fused rotary embedding for q [B,T,Hq,D] and k [B,T,Hk,D]",
        py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"));
    m.def("swiglu", &swiglu,
        "silu(gate) * up",
        py::arg("gate"), py::arg("up"));
    m.def("rmsnorm", &rmsnorm,
        "input: (x: torch.Tensor(..., D, FP16, CUDA), eps: float, mean_dim: int)\n"
        "output: torch.Tensor(..., D, FP16)",
        py::arg("x"), py::arg("eps"), py::arg("mean_dim"));
    m.def("sum_i4", &sum_i4,
        "input: (x: torch.Tensor(M x K_packed, UINT8, CUDA), sym: bool)\n"
        "output: torch.Tensor(M x 1, INT32, CUDA)\n"
        "output = row-wise sum over unpacked int4 values without explicit unpacking",
        py::arg("x"), py::arg("sym") = true);
    m.def("matmul_colrow_i8_i32", &matmul_colrow_i8_i32,
        "input: (a_col: torch.Tensor(N x 1, INT8, CUDA), b_row: torch.Tensor(1 x C, INT32, CUDA))\n"
        "output: torch.Tensor(N x C, INT32, CUDA)",
        py::arg("a_col"), py::arg("b_row"));
    m.def("batch_decode_i4", &batch_decode_i4,
          "int4 paged batch decode; pass a [batch, heads, chunks, head_dim+4] fp32 "
          "workspace and num_chunks > 1 to split each sequence across blocks",
          py::arg("o"), py::arg("q"), py::arg("kv_data"), py::arg("kv_param"),
          py::arg("kv_indptr"), py::arg("kv_indices"), py::arg("last_page_offset"),
          py::arg("layer_idx"), py::arg("tmp") = std::nullopt,
          py::arg("num_chunks") = 1);
    m.def("init_kv_i4", &init_kv_i4, "");
    m.def("append_kv_i4", &append_kv_i4, "");
    m.def("batch_decode_f16", &batch_decode_f16, "");
    m.def("init_kv_f16", &init_kv_f16, "");
    m.def("append_kv_f16", &append_kv_f16, ""); 

}
