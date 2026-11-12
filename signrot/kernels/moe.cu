// Expert-indexed fused kernels for the MoE block (see moe.h).
//
// These mirror the fused.cu activation quantizers op-for-op -- the only
// addition is the per-row expert indirection into the stacked sign /
// weight-scale / weight-sum tables, so the codes, scales and dequant results
// stay bit-identical to the per-expert kernel chain they replace.

#include <stream.h>
#include <moe.h>
#include <cfloat>


// same reduction helpers as fused.cu (kept file-local there)

DEVICE float moe_warp_reduce_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

DEVICE float moe_warp_reduce_min(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fminf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

template <int kBlock, int kOp>
DEVICE float moe_block_reduce(float v, float *smem, float ident) {
    constexpr int kWarps = kBlock / 32;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = (kOp == 0) ? moe_warp_reduce_max(v) : moe_warp_reduce_min(v);

    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < kWarps) ? smem[lane] : ident;
        v = (kOp == 0) ? moe_warp_reduce_max(v) : moe_warp_reduce_min(v);
        if (lane == 0) smem[kWarps] = v;
    }
    __syncthreads();
    float r = smem[kWarps];
    __syncthreads();
    return r;
}

// fp16 * (+-1) is exact, so flipping the sign bit reproduces the eager
// `x * had_adj_sign` multiply bit-for-bit
DEVICE half signed_load(const half *xr, const int8_t *sign_row, uint32_t i) {
    half v = xr[i];
    if (sign_row != nullptr && sign_row[i] < 0) v = __hneg(v);
    return v;
}


template <int kBlock>
__global__ void sym_quant_dynamic_moe_kernel(
    const half *__restrict__ x,
    const int8_t *__restrict__ sign,
    const int32_t *__restrict__ expert_ids,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio,
    half *__restrict__ scale_out,
    Int4Storage *__restrict__ q)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * colsSrc;
    const int8_t *sr = sign ? sign + (size_t)expert_ids[row] * colsSrc : nullptr;

    float amax = 0.f;
    for (uint32_t i = threadIdx.x; i < colsSrc; i += kBlock)
        amax = fmaxf(amax, fabsf(__half2float(signed_load(xr, sr, i))));

    __shared__ float smem[kBlock / 32 + 1];
    amax = moe_block_reduce<kBlock, 0>(amax, smem, 0.f);

    half scale = __float2half(amax / 7.f);
    if (clip_ratio != 1.f) scale = __float2half(__half2float(scale) * clip_ratio);
    if (threadIdx.x == 0) scale_out[row] = scale;

    Int4Storage *qr = q + (size_t)row * colsDst;
    for (uint32_t colDst = threadIdx.x; colDst < colsDst; colDst += kBlock) {
        Int4Storage storage;
        memset(&storage, 0, sizeof(storage));
        const uint32_t id = colDst * kElementsPerVector;
#pragma unroll
        for (int i = 0; i < (int)kElementsPerVector; ++i) {
            if (id + i < colsSrc) {
                half data = __hdiv(signed_load(xr, sr, id + i), scale);
                int qval = clamp(__half2int_rn(data), qmin_si4, qmax_si4);
                Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
            }
        }
        qr[colDst] = storage;
    }
}


template <int kBlock>
__global__ void asym_quant_dynamic_moe_kernel(
    const half *__restrict__ x,
    const int8_t *__restrict__ sign,
    const int32_t *__restrict__ expert_ids,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio,
    half *__restrict__ scale_out,
    half *__restrict__ zero_out,
    UInt4Storage *__restrict__ q)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * colsSrc;
    const int8_t *sr = sign ? sign + (size_t)expert_ids[row] * colsSrc : nullptr;

    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (uint32_t i = threadIdx.x; i < colsSrc; i += kBlock) {
        float v = __half2float(signed_load(xr, sr, i));
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }
    __shared__ float smax[kBlock / 32 + 1];
    __shared__ float smin[kBlock / 32 + 1];
    vmax = moe_block_reduce<kBlock, 0>(vmax, smax, -FLT_MAX);
    vmin = moe_block_reduce<kBlock, 1>(vmin, smin, FLT_MAX);

    half hmin = __float2half(vmin), hmax = __float2half(vmax);
    if (clip_ratio != 1.f) {
        hmin = __float2half(__half2float(hmin) * clip_ratio);
        hmax = __float2half(__half2float(hmax) * clip_ratio);
    }
    half diff = __float2half(__half2float(hmax) - __half2float(hmin));
    half scale = __float2half(__half2float(diff) / (float)qmax_ui4);
    half ratio = __float2half(-__half2float(hmin) / __half2float(scale));
    half zero = __float2half(nearbyintf(__half2float(ratio)));
    if (threadIdx.x == 0) {
        scale_out[row] = scale;
        zero_out[row] = zero;
    }
    const int zero_i = __half2int_rn(zero);

    UInt4Storage *qr = q + (size_t)row * colsDst;
    for (uint32_t colDst = threadIdx.x; colDst < colsDst; colDst += kBlock) {
        UInt4Storage storage;
        memset(&storage, 0, sizeof(storage));
        const uint32_t id = colDst * kElementsPerVectorUInt4;
#pragma unroll
        for (int i = 0; i < (int)kElementsPerVectorUInt4; ++i) {
            if (id + i < colsSrc) {
                half data = __hdiv(signed_load(xr, sr, id + i), scale);
                int qval = clamp(__half2int_rn(data) + zero_i, qmin_ui4, qmax_ui4);
                UInt4Subbyte(reinterpret_cast<cutlass::uint4b_t *>(&storage), i).set(qval);
            }
        }
        qr[colDst] = storage;
    }
}


// same per-row block sizing policy as fused.cu
static HOST int moe_pick_block(uint32_t rows, uint32_t cols) {
    if (cols <= 128) return 128;
    if (cols <= 512) return 256;
    return rows < 64 ? 1024 : 256;
}

#define MOE_DISPATCH_BLOCK(rows, cols, KERNEL, ...)                        \
    do {                                                                   \
        if ((rows) == 0) break;                                            \
        switch (moe_pick_block((rows), (cols))) {                          \
            case 128: KERNEL<128><<<(rows), 128, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break;    \
            case 1024: KERNEL<1024><<<(rows), 1024, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break; \
            default: KERNEL<256><<<(rows), 256, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break;     \
        }                                                                  \
    } while (0)

void sym_quant_dynamic_moe_host(
    const half *x, const int8_t *sign, const int32_t *expert_ids,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio, half *scale, Int4Storage *q)
{
    MOE_DISPATCH_BLOCK(rows, colsSrc, sym_quant_dynamic_moe_kernel,
                       x, sign, expert_ids, rows, colsSrc, colsDst, clip_ratio, scale, q);
}

void asym_quant_dynamic_moe_host(
    const half *x, const int8_t *sign, const int32_t *expert_ids,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio, half *scale, half *zero, UInt4Storage *q)
{
    MOE_DISPATCH_BLOCK(rows, colsSrc, asym_quant_dynamic_moe_kernel,
                       x, sign, expert_ids, rows, colsSrc, colsDst, clip_ratio, scale, zero, q);
}


// --------------------------------------------- expert-indexed dequant

__global__ void sym_dequant_moe_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ w_scales,
        const int32_t *__restrict__ expert_ids,
        const int8_t *__restrict__ zero_row,
        const int32_t *__restrict__ w_sum,
        uint32_t rows, uint32_t cols,
        half *__restrict__ out)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= cols || row >= rows) return;

    const size_t ecol = (size_t)expert_ids[row] * cols + col;
    int32_t qval = q[col + (size_t)row * cols];
    if (zero_row != nullptr)
        qval -= (int32_t)zero_row[row] * w_sum[ecol];  // exact integer correction

    // identical arithmetic to quant.cu sym_dequant_i32_f16_kernel
    float qf = static_cast<float>(qval);
    float row_f = __half2float(scale_row[row]);
    float col_f = __half2float(w_scales[ecol]);
    out[col + (size_t)row * cols] = __float2half_rn(qf * row_f * col_f);
}

void sym_dequant_moe_host(
    const int32_t *q, const half *scale_row, const half *w_scales,
    const int32_t *expert_ids, const int8_t *zero_row, const int32_t *w_sum,
    uint32_t rows, uint32_t cols, half *out)
{
    if (rows == 0) return;
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequant_moe_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(
            q, scale_row, w_scales, expert_ids, zero_row, w_sum, rows, cols, out);
}
