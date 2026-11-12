// Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
#pragma once

#include <common.h>


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
);


void sym_dequant_host(
        const int32_t *q,
        const half *scale_row,
        const half *scale_col,
        uint32_t rows,
        uint32_t cols,
        half *x
);


void asym_quant_f16_si4_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);


void asym_quant_f16_ui4_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    UInt4Storage *q
);


void asym_batch_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *zeros_row,
    const half *scale_col,
    const half *zeros_col,
    uint32_t batches,
    uint32_t rows,
    uint32_t cols,
    half *x
);


void sum_i4_host(
    const uint8_t *x,
    uint32_t rows,
    uint32_t packed_cols,
    bool is_signed,
    int32_t *out
);


void matmul_colrow_i8_i32_host(
    const int8_t *a_col,
    const int32_t *b_row,
    uint32_t rows,
    uint32_t cols,
    int32_t *out
);
