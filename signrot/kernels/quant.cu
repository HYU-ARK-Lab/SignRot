// Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
#include <stream.h>
#include <quant.h>


template<typename T>
__device__ __half int_to_half(T value)
{
    return __int2half_rn(static_cast<int>(value));
}


__global__
void sym_quant_f16_i4_kernel(
        const half *__restrict__ x,
        const half *__restrict__ scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);

            int qval = clamp(__half2int_rn(data), qmin_si4, qmax_si4);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
                    qval);
        }
    }

    q[colDst + row * colsDst] = storage;
}


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
)
{

    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quant_f16_i4_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(x, scale, rows, colsSrc, colsDst, q);
}


__global__ void sym_dequant_i32_f16_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ scale_col,
        uint32_t rows, uint32_t cols,
        half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    // half xElement = int_to_half(q[col + row * cols]);
    // x[col + row * cols] = scale_row[row] * scale_col[col] * xElement;

    float qf = static_cast<float>(q[col + row * cols]);
    float row_f = __half2float(scale_row[row]);
    float col_f = __half2float(scale_col[col]);
    x[col + row * cols] = __float2half_rn(qf * row_f * col_f);

    // int32_t qval = q[col + row * cols];
    // half row_scale = scale_row[row];
    // half col_scale = scale_col[col];

    // if constexpr (kMode == 0)
    // {
    //     // Original half math path: (scale_row * scale_col) * q
    //     half xElement = int_to_half(qval);
    //     x[col + row * cols] = row_scale * col_scale * xElement;
    // }
    // else if constexpr (kMode == 1)
    // {
    //     // Overflow-avoid split path: q = 2 * (q >> 1) + (q & 1)
    //     half xElement = int_to_half(qval >> 1);
    //     half xResidual = int_to_half(qval & 1);
    //     x[col + row * cols] = (xElement * row_scale) * __hmul(col_scale, int_to_half(2));
    //     x[col + row * cols] += xResidual * row_scale * col_scale;
    // }
    // else
    // {
    //     // Float accumulation path: compute in fp32 then cast to fp16
    //     float qf = static_cast<float>(qval);
    //     float row_f = __half2float(row_scale);
    //     float col_f = __half2float(col_scale);
    //     x[col + row * cols] = __float2half_rn(qf * row_f * col_f);
    // }
}

void sym_dequant_host(const int32_t *q,
                                 const half *scale_row,
                                 const half *scale_col,
                                 uint32_t rows,
                                 uint32_t cols,
                                 half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequant_i32_f16_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(
            q,
            scale_row, scale_col,
            rows, cols, x);
}

__global__
void asym_quant_f16_si4_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale,
    const half *__restrict__ zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);
            int qval = clamp(__half2int_rn(data) + __half2int_rn(zeros[row]), qmin_si4, qmax_si4);
            Int4Subbyte(reinterpret_cast<cutlass::int4b_t *>(&storage), i).set(qval);
        }
    }
    
    q[colDst + row * colsDst] = storage;
}

void asym_quant_f16_si4_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
)
{
    dim3 block{std::min<uint32_t>(colsDst, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    asym_quant_f16_si4_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(x, scale, zeros, rows, colsSrc, colsDst, q);
}

__global__
void asym_quant_f16_ui4_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale,
    const half *__restrict__ zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    UInt4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVectorUInt4 >= colsSrc)
    {
        return;
    }
    UInt4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVectorUInt4 + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVectorUInt4; ++i)
    {
        bool safe = (colDst * kElementsPerVectorUInt4 + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);
            int qval = clamp(__half2int_rn(data) + __half2int_rn(zeros[row]), qmin_ui4, qmax_ui4);
            UInt4Subbyte(reinterpret_cast<cutlass::uint4b_t *>(&storage), i).set(qval);
        }
    }
    
    q[colDst + row * colsDst] = storage;
}

void asym_quant_f16_ui4_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    UInt4Storage *q
)
{
    dim3 block{std::min<uint32_t>(colsDst, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    asym_quant_f16_ui4_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(x, scale, zeros, rows, colsSrc, colsDst, q);
}

template <bool kSigned>
__global__ void sum_i4_kernel(
    const uint8_t *__restrict__ x,
    uint32_t rows,
    uint32_t packed_cols,
    int32_t *__restrict__ out
)
{
    uint32_t row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kMaxWarpsPerBlock = 8; // for blockDim.x <= 256
    uint32_t tid = threadIdx.x;

    int local_sum = 0;
    const uint8_t *row_ptr = x + row * packed_cols;

    bool aligned_u32 = (reinterpret_cast<uintptr_t>(row_ptr) & (sizeof(uint32_t) - 1)) == 0;
    uint32_t vec_cols = packed_cols / 4;
    uint32_t rem_start = vec_cols * 4;

    if (aligned_u32)
    {
        const uint32_t *row_ptr_u32 = reinterpret_cast<const uint32_t *>(row_ptr);
        for (uint32_t col4 = tid; col4 < vec_cols; col4 += blockDim.x)
        {
            uint32_t word = row_ptr_u32[col4];

#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint32_t byte = (word >> (j * 8)) & 0xFF;
                int low = static_cast<int>(byte & 0x0F);
                int high = static_cast<int>((byte >> 4) & 0x0F);

                if constexpr (kSigned)
                {
                    low = ((low + 8) & 0x0F) - 8;
                    high = ((high + 8) & 0x0F) - 8;
                }

                local_sum += (low + high);
            }
        }
    }

    uint32_t scalar_start = aligned_u32 ? rem_start : 0;
    for (uint32_t col = scalar_start + tid; col < packed_cols; col += blockDim.x)
    {
        uint8_t v = row_ptr[col];
        int low = static_cast<int>(v & 0x0F);
        int high = static_cast<int>((v >> 4) & 0x0F);

        if constexpr (kSigned)
        {
            low = ((low + 8) & 0x0F) - 8;
            high = ((high + 8) & 0x0F) - 8;
        }

        local_sum += (low + high);
    }

    for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    __shared__ int warp_sums[kMaxWarpsPerBlock];
    uint32_t lane = tid & (kWarpSize - 1);
    uint32_t warp = tid / kWarpSize;
    uint32_t warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

    if (lane == 0)
    {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0)
    {
        int block_sum = (lane < warp_count) ? warp_sums[lane] : 0;
        for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }

        if (lane == 0)
        {
            out[row] = block_sum;
        }
    }
}

void sum_i4_host(
    const uint8_t *x,
    uint32_t rows,
    uint32_t packed_cols,
    bool is_signed,
    int32_t *out
)
{
    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);

    if (is_signed)
    {
        sum_i4_kernel<true><<<grid, block, 0, SIGNROT_STREAM>>>(x, rows, packed_cols, out);
    }
    else
    {
        sum_i4_kernel<false><<<grid, block, 0, SIGNROT_STREAM>>>(x, rows, packed_cols, out);
    }
}

__global__ void matmul_colrow_i8_i32_kernel(
    const int8_t *__restrict__ a_col,
    const int32_t *__restrict__ b_row,
    uint32_t rows,
    uint32_t cols,
    int32_t *__restrict__ out
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= rows || col >= cols)
    {
        return;
    }

    int32_t a = static_cast<int32_t>(a_col[row]);
    out[row * cols + col] = a * b_row[col];
}

void matmul_colrow_i8_i32_host(
    const int8_t *a_col,
    const int32_t *b_row,
    uint32_t rows,
    uint32_t cols,
    int32_t *out
)
{
    dim3 block{32, 8};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    matmul_colrow_i8_i32_kernel<<<grid, block, 0, SIGNROT_STREAM>>>(a_col, b_row, rows, cols, out);
}
