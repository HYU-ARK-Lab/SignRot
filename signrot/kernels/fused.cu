// Fused pre-GEMM kernels for the decode path.
//
// The deployment graph used to spend most of its time launching tiny elementwise
// kernels: a single activation quantization was amax + abs + div + cast + mul +
// quant (6 launches) and the KV-cache quantize+pack was ~15.  Each of these
// fuses the whole chain into one launch while keeping the arithmetic
// bit-identical to the PyTorch/CUDA sequence it replaces (see fused.h).

#include <stream.h>
#include <fused.h>
#include <cfloat>


// ---------------------------------------------------------------- reductions

DEVICE float warp_reduce_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

DEVICE float warp_reduce_min(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fminf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

DEVICE float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Block-wide reduction returning the result to every thread.  `smem` must hold
// at least kBlock/32 floats plus one broadcast slot.
template <int kBlock, int kOp>
DEVICE float block_reduce(float v, float *smem, float ident) {
    constexpr int kWarps = kBlock / 32;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    if (kOp == 0) v = warp_reduce_max(v);
    else if (kOp == 1) v = warp_reduce_min(v);
    else v = warp_reduce_sum(v);

    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < kWarps) ? smem[lane] : ident;
        if (kOp == 0) v = warp_reduce_max(v);
        else if (kOp == 1) v = warp_reduce_min(v);
        else v = warp_reduce_sum(v);
        if (lane == 0) smem[kWarps] = v;
    }
    __syncthreads();
    float r = smem[kWarps];
    __syncthreads();
    return r;
}


// ------------------------------------------------- symmetric activation quant

template <int kBlock>
__global__ void sym_quant_dynamic_kernel(
    const half *__restrict__ x,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio,
    half *__restrict__ scale_out,
    Int4Storage *__restrict__ q)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * colsSrc;

    // torch.max(torch.abs(x), dim=-1): max over fp16 magnitudes.  Reducing in
    // fp32 is exact here because every input is fp16-representable and max
    // never rounds.
    float amax = 0.f;
    for (uint32_t i = threadIdx.x; i < colsSrc; i += kBlock)
        amax = fmaxf(amax, fabsf(__half2float(xr[i])));

    __shared__ float smem[kBlock / 32 + 1];
    amax = block_reduce<kBlock, 0>(amax, smem, 0.f);

    // (max_val / 7) rounds to fp16, then the clip_ratio multiply rounds again;
    // both are no-ops in the common clip_ratio == 1 case but are kept so the
    // result matches Quantizer.forward exactly.
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
                half data = __hdiv(xr[id + i], scale);
                int qval = clamp(__half2int_rn(data), qmin_si4, qmax_si4);
                Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
            }
        }
        qr[colDst] = storage;
    }
}

// Short rows (head_dim) do not need a wide block; few rows (batch-1 decode)
// benefit from as many threads per row as possible.
HOST int pick_block(uint32_t rows, uint32_t cols) {
    if (cols <= 128) return 128;
    if (cols <= 512) return 256;
    return rows < 64 ? 1024 : 256;
}

#define DISPATCH_BLOCK(rows, cols, KERNEL, ...)                            \
    do {                                                                   \
        if ((rows) == 0) break;                                            \
        switch (pick_block((rows), (cols))) {                              \
            case 128: KERNEL<128><<<(rows), 128, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break;    \
            case 1024: KERNEL<1024><<<(rows), 1024, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break; \
            default: KERNEL<256><<<(rows), 256, 0, SIGNROT_STREAM>>>(__VA_ARGS__); break;     \
        }                                                                  \
    } while (0)

void sym_quant_dynamic_host(
    const half *x, uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio, half *scale, Int4Storage *q)
{
    DISPATCH_BLOCK(rows, colsSrc, sym_quant_dynamic_kernel,
                   x, rows, colsSrc, colsDst, clip_ratio, scale, q);
}


// ------------------------------------------------ asymmetric activation quant

template <int kBlock>
__global__ void asym_quant_dynamic_kernel(
    const half *__restrict__ x,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio,
    half *__restrict__ scale_out,
    half *__restrict__ zero_out,
    UInt4Storage *__restrict__ q)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * colsSrc;

    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (uint32_t i = threadIdx.x; i < colsSrc; i += kBlock) {
        float v = __half2float(xr[i]);
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }
    __shared__ float smax[kBlock / 32 + 1];
    __shared__ float smin[kBlock / 32 + 1];
    vmax = block_reduce<kBlock, 0>(vmax, smax, -FLT_MAX);
    vmin = block_reduce<kBlock, 1>(vmin, smin, FLT_MAX);

    // x.amin(...) * clip_ratio and x.amax(...) * clip_ratio each round to fp16
    half hmin = __float2half(vmin), hmax = __float2half(vmax);
    if (clip_ratio != 1.f) {
        hmin = __float2half(__half2float(hmin) * clip_ratio);
        hmax = __float2half(__half2float(hmax) * clip_ratio);
    }
    // (max - min) rounds to fp16, then / 15 rounds again
    half diff = __float2half(__half2float(hmax) - __half2float(hmin));
    half scale = __float2half(__half2float(diff) / (float)qmax_ui4);
    // (-min / scale) rounds to fp16 before .round() is applied
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
                half data = __hdiv(xr[id + i], scale);
                int qval = clamp(__half2int_rn(data) + zero_i, qmin_ui4, qmax_ui4);
                UInt4Subbyte(reinterpret_cast<cutlass::uint4b_t *>(&storage), i).set(qval);
            }
        }
        qr[colDst] = storage;
    }
}

void asym_quant_dynamic_host(
    const half *x, uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    float clip_ratio, half *scale, half *zero, UInt4Storage *q)
{
    DISPATCH_BLOCK(rows, colsSrc, asym_quant_dynamic_kernel,
                   x, rows, colsSrc, colsDst, clip_ratio, scale, zero, q);
}


// -------------------------------------------------------- KV quantize + pack

// One block per (batch, token, head) row; the row length is head_dim (128).
template <int kBlock>
__global__ void kv_quant_pack_kernel(
    const half *__restrict__ x,
    uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    half *__restrict__ param,
    uint8_t *__restrict__ q)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * colsSrc;

    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (uint32_t i = threadIdx.x; i < colsSrc; i += kBlock) {
        float v = __half2float(xr[i]);
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }
    __shared__ float smax[kBlock / 32 + 1];
    __shared__ float smin[kBlock / 32 + 1];
    vmax = block_reduce<kBlock, 0>(vmax, smax, -FLT_MAX);
    vmin = block_reduce<kBlock, 1>(vmin, smin, FLT_MAX);

    half hmin = __float2half(vmin), hmax = __float2half(vmax);
    // (xmax - xmin).clamp(min=1e-5) / 15
    half diff = __float2half(__half2float(hmax) - __half2float(hmin));
    diff = __float2half(fmaxf(__half2float(diff), 1e-5f));
    half scale = __float2half(__half2float(diff) / (float)qmax_ui4);
    half zero = __hneg(hmin);
    if (threadIdx.x == 0) {
        param[row * 2 + 0] = scale;
        param[row * 2 + 1] = zero;
    }

    const float fscale = __half2float(scale), fzero = __half2float(zero);
    uint8_t *qr = q + (size_t)row * colsDst;
    for (uint32_t colDst = threadIdx.x; colDst < colsDst; colDst += kBlock) {
        const uint32_t id = colDst * 2;
        uint8_t packed = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (id + i < colsSrc) {
                // (x + zero) rounds to fp16, / scale rounds to fp16, round(),
                // clamp(0, 15) -- the same op-by-op sequence as the torch.jit
                // implementation this replaces
                half shifted = __float2half(__half2float(xr[id + i]) + fzero);
                half scaled = __float2half(__half2float(shifted) / fscale);
                int qval = clamp((int)nearbyintf(__half2float(scaled)), qmin_ui4, qmax_ui4);
                packed |= (uint8_t)(qval << (4 * i));
            }
        }
        qr[colDst] = packed;
    }
}

void kv_quant_pack_host(
    const half *x, uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
    half *param, uint8_t *q)
{
    DISPATCH_BLOCK(rows, colsSrc, kv_quant_pack_kernel, x, rows, colsSrc, colsDst, param, q);
}


// ------------------------------------------------------------------ RMSNorm

template <int kBlock>
__global__ void rmsnorm_kernel(
    const half *__restrict__ x,
    uint32_t rows, uint32_t cols, float eps, float mean_dim,
    half *__restrict__ out)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const half *xr = x + (size_t)row * cols;

    float acc = 0.f;
    for (uint32_t i = threadIdx.x; i < cols; i += kBlock) {
        float v = __half2float(xr[i]);
        acc += v * v;
    }
    __shared__ float smem[kBlock / 32 + 1];
    acc = block_reduce<kBlock, 2>(acc, smem, 0.f);

    // variance = sum / mean_dim, then rsqrt(variance + eps) -- kept as a
    // division so non-power-of-two mean_dim matches the torch reference
    const float inv = rsqrtf(acc / mean_dim + eps);
    half *orow = out + (size_t)row * cols;
    for (uint32_t i = threadIdx.x; i < cols; i += kBlock)
        orow[i] = __float2half(__half2float(xr[i]) * inv);
}

void rmsnorm_host(
    const half *x, uint32_t rows, uint32_t cols, float eps,
    uint32_t mean_dim, half *out)
{
    DISPATCH_BLOCK(rows, cols, rmsnorm_kernel, x, rows, cols, eps, (float)mean_dim, out);
}


// ------------------------------------------------------------ rotary (RoPE)

// One block per (token, head); each thread owns the (i, i + head_dim/2) pair
// that rotate_half couples, so the update is race-free in place.
__global__ void rope_qk_kernel(
    half *__restrict__ q,
    half *__restrict__ k,
    const half *__restrict__ cos,
    const half *__restrict__ sin,
    uint32_t seq_len, uint32_t cos_seqs,
    uint32_t n_q_heads, uint32_t n_k_heads, uint32_t head_dim)
{
    const uint32_t half_dim = head_dim >> 1;
    const uint32_t total_heads = n_q_heads + n_k_heads;
    const uint32_t token = blockIdx.x / total_heads;
    const uint32_t head = blockIdx.x - token * total_heads;

    half *base = (head < n_q_heads)
        ? q + ((size_t)token * n_q_heads + head) * head_dim
        : k + ((size_t)token * n_k_heads + (head - n_q_heads)) * head_dim;

    // cos/sin carry a leading batch dim of 1 when they broadcast
    const uint32_t cos_token = (cos_seqs == 1) ? (token % seq_len) : token;
    const half *c = cos + (size_t)cos_token * head_dim;
    const half *s = sin + (size_t)cos_token * head_dim;

    for (uint32_t i = threadIdx.x; i < half_dim; i += blockDim.x) {
        const float x0 = __half2float(base[i]);
        const float x1 = __half2float(base[i + half_dim]);
        // (x * cos) and (rotate_half(x) * sin) each round to fp16 before the
        // add, matching the three-op torch expression
        const half a0 = __float2half(x0 * __half2float(c[i]));
        const half b0 = __float2half(-x1 * __half2float(s[i]));
        const half a1 = __float2half(x1 * __half2float(c[i + half_dim]));
        const half b1 = __float2half(x0 * __half2float(s[i + half_dim]));
        base[i] = __float2half(__half2float(a0) + __half2float(b0));
        base[i + half_dim] = __float2half(__half2float(a1) + __half2float(b1));
    }
}

void rope_qk_host(
    half *q, half *k, const half *cos, const half *sin,
    uint32_t n_seqs, uint32_t seq_len, uint32_t cos_seqs,
    uint32_t n_q_heads, uint32_t n_k_heads, uint32_t head_dim)
{
    const uint32_t blocks = n_seqs * seq_len * (n_q_heads + n_k_heads);
    if (blocks == 0) return;
    const uint32_t threads = mymin<uint32_t>(head_dim / 2, 256u);
    rope_qk_kernel<<<blocks, threads, 0, SIGNROT_STREAM>>>(
        q, k, cos, sin, seq_len, cos_seqs, n_q_heads, n_k_heads, head_dim);
}


// ------------------------------------------------------------------ SwiGLU

__global__ void swiglu_kernel(
    const half *__restrict__ gate,
    const half *__restrict__ up,
    uint32_t numel,
    half *__restrict__ out)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (; i < numel; i += stride) {
        // torch SiLU accumulates in fp32 and rounds once, then the multiply
        // rounds again
        const float g = __half2float(gate[i]);
        const half s = __float2half(g / (1.f + expf(-g)));
        out[i] = __float2half(__half2float(s) * __half2float(up[i]));
    }
}

void swiglu_host(const half *gate, const half *up, uint32_t numel, half *out)
{
    if (numel == 0) return;
    const uint32_t threads = 256;
    const uint32_t blocks = mymin<uint32_t>(cdiv(numel, threads), 65535u);
    swiglu_kernel<<<blocks, threads, 0, SIGNROT_STREAM>>>(gate, up, numel, out);
}
