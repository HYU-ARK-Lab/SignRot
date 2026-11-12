#pragma once

#include <common.h>


// Dynamic per-row symmetric int4 activation quantization.
// Fuses  amax -> scale -> divide/round/clamp -> int4 pack  into one kernel,
// reproducing exactly:
//     max_val  = torch.max(torch.abs(x), dim=-1)[0]
//     scale    = (max_val / 7).to(fp16) * clip_ratio
//     q        = sym_quant(x, scale)
void sym_quant_dynamic_host(
    const half *x,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    float clip_ratio,
    half *scale,
    Int4Storage *q
);


// Dynamic per-row asymmetric uint4 activation quantization, reproducing:
//     min_val = x.amin(-1) * clip_ratio ; max_val = x.amax(-1) * clip_ratio
//     scale   = (max_val - min_val) / 15
//     zero    = (-min_val / scale).round()
//     q       = asym_quant_f16_ui4(x, scale, zero)
void asym_quant_dynamic_host(
    const half *x,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    float clip_ratio,
    half *scale,
    half *zero,
    UInt4Storage *q
);


// Fused KV-cache quantize+pack, reproducing asym_quantize_and_pack_i4(lac=False):
//     xmax = amax(x, -1) ; xmin = amin(x, -1)
//     scale = (xmax - xmin).clamp(min=1e-5) / 15
//     zero  = -xmin
//     q     = clamp(round((x + zero) / scale), 0, 15) packed two-per-byte
// `param` is written as the [scale, zero] pair per row, i.e. what the caller
// used to build with torch.cat.
void kv_quant_pack_host(
    const half *x,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    half *param,
    uint8_t *q
);


// Fused rotary embedding for Q and K in place, reproducing HF's
// apply_rotary_pos_emb(..., unsqueeze_dim=2):
//     out = x * cos + rotate_half(x) * sin
// `cos`/`sin` are [cos_seqs, seq_len, head_dim]; cos_seqs may be 1, in which
// case they broadcast across the batch as they do in the torch version.
void rope_qk_host(
    half *q,
    half *k,
    const half *cos,
    const half *sin,
    uint32_t n_seqs,
    uint32_t seq_len,
    uint32_t cos_seqs,
    uint32_t n_q_heads,
    uint32_t n_k_heads,
    uint32_t head_dim
);


// Fused SwiGLU: silu(gate) * up, reproducing torch.nn.SiLU()(gate) * up.
void swiglu_host(
    const half *gate,
    const half *up,
    uint32_t numel,
    half *out
);


// Fused RMSNorm over the last dimension (fp32 accumulation, fp16 in/out),
// reproducing signrot.nn.RMSNorm.
void rmsnorm_host(
    const half *x,
    uint32_t rows,
    uint32_t cols,
    float eps,
    uint32_t mean_dim,
    half *out
);
