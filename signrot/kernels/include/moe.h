#pragma once

#include <common.h>

// Expert-indexed variants of the fused activation quantizers and the dequant
// for the MoE block: rows are token copies sorted by expert, `expert_ids[row]`
// selects each row's expert in the stacked [num_experts, cols] tables.


// sym_quant_dynamic with an optional per-expert had_adj sign flip fused in
// front, reproducing exactly:
//     x     = x * sign[expert_ids[row]]        (skipped when sign == nullptr)
//     scale = (x.abs().amax(-1) / 7).to(fp16) * clip_ratio
//     q     = sym_quant(x, scale)
// Multiplying fp16 by +-1 is exact, so the fused flip is bit-identical to the
// separate elementwise multiply it replaces.
void sym_quant_dynamic_moe_host(
    const half *x,
    const int8_t *sign,        // [num_experts, colsSrc] or nullptr
    const int32_t *expert_ids, // [rows]
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    float clip_ratio,
    half *scale,
    Int4Storage *q
);


// asym_quant_dynamic with the same optional fused sign flip.
void asym_quant_dynamic_moe_host(
    const half *x,
    const int8_t *sign,
    const int32_t *expert_ids,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    float clip_ratio,
    half *scale,
    half *zero,
    UInt4Storage *q
);


// Expert-indexed dequant with the asym zero-point correction fused in,
// reproducing (per row, e = expert_ids[row]):
//     q  -= zero_row[row] * w_sum[e]        (int32, skipped when nullptr)
//     out = q * scale_row[row] * w_scales[e]   (fp32 math, fp16 store)
// i.e. matmul_colrow_i8_i32 + the int32 subtract + sym_dequant in one kernel.
void sym_dequant_moe_host(
    const int32_t *q,
    const half *scale_row,     // [rows]
    const half *w_scales,      // [num_experts, cols]
    const int32_t *expert_ids, // [rows]
    const int8_t *zero_row,    // [rows] or nullptr
    const int32_t *w_sum,      // [num_experts, cols] or nullptr
    uint32_t rows,
    uint32_t cols,
    half *out
);
