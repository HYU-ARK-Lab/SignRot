"""Numerical equivalence of the fused decode kernels against the paths they replace.

The quantizers must be bit-exact: they decide 4-bit codes, so any drift there is
an accuracy change.  RMSNorm accumulates in fp32 in a different order than the
torch reduction, so it is checked to be within one fp16 ulp.

Run: .venv/bin/python benchmarks/test_fused_equivalence.py
"""

import torch

import signrot.functional.kernel as kernel
import signrot.nn
from signrot.transformers.kv_cache import asym_quantize_and_pack_i4

DEV = "cuda"
FAILED = []


def check(name, got, ref, exact=True, atol=0.0):
    if exact:
        ok = torch.equal(got, ref)
        detail = "bit-exact" if ok else f"{(got != ref).sum().item()} / {ref.numel()} differ"
    else:
        diff = (got.float() - ref.float()).abs()
        ok = bool((diff <= atol).all())
        detail = f"max abs diff {diff.max().item():.3e} (tol {atol:.3e})"
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:52s} {detail}")
    if not ok:
        FAILED.append(name)


def shapes():
    # decode (M=1..16) and prefill (M=2048) row counts x the Llama-2-7B widths
    for rows in (1, 8, 16, 2048):
        for cols in (4096, 11008, 128):
            yield rows, cols


print("=== symmetric activation quantizer (nn.Quantizer) ===")
q = signrot.nn.Quantizer()
for rows, cols in shapes():
    x = (torch.randn(1, rows, cols, device=DEV, dtype=torch.float16) * 3)
    got = q(x)
    ref = q.forward_reference(x)
    check(f"sym rows={rows} cols={cols} q", got.quantized_x, ref.quantized_x)
    check(f"sym rows={rows} cols={cols} scale",
          got.scales_x.reshape(-1), ref.scales_x.reshape(-1))

print("\n=== symmetric quantizer, edge cases ===")
for name, x in [
    ("all zeros", torch.zeros(1, 4, 4096, device=DEV, dtype=torch.float16)),
    ("single spike", torch.zeros(1, 4, 4096, device=DEV, dtype=torch.float16).index_put_(
        (torch.tensor([0]), torch.tensor([1]), torch.tensor([7])),
        torch.tensor([100.0], device=DEV, dtype=torch.float16))),
    ("large dynamic range",
     (torch.randn(1, 4, 4096, device=DEV, dtype=torch.float16) * 1e3)),
    ("tiny values", (torch.randn(1, 4, 4096, device=DEV, dtype=torch.float16) * 1e-4)),
    ("all negative", -(torch.rand(1, 4, 4096, device=DEV, dtype=torch.float16) + 1)),
]:
    got, ref = q(x), q.forward_reference(x)
    check(f"sym {name} q", got.quantized_x, ref.quantized_x)
    check(f"sym {name} scale", got.scales_x.reshape(-1), ref.scales_x.reshape(-1))

print("\n=== asymmetric activation quantizer (nn.AsymQuantizer) ===")
aq = signrot.nn.AsymQuantizer()
for rows, cols in shapes():
    x = torch.randn(1, rows, cols, device=DEV, dtype=torch.float16) * 3
    got, ref = aq(x), aq.forward_reference(x)
    check(f"asym rows={rows} cols={cols} q", got.quantized_x, ref.quantized_x)
    check(f"asym rows={rows} cols={cols} scale",
          got.scales_x.reshape(-1), ref.scales_x.reshape(-1))
    check(f"asym rows={rows} cols={cols} zero",
          got.zeros_x.reshape(-1), ref.zeros_x.reshape(-1))

print("\n=== KV quantize + pack (kv_quant_pack vs asym_quantize_and_pack_i4) ===")
for b, t, h, d in ((1, 1, 32, 128), (1, 2048, 32, 128), (16, 1, 32, 128), (4, 37, 8, 128)):
    x = torch.randn(b, t, h, d, device=DEV, dtype=torch.float16) * 2
    q_f, param = kernel.kv_quant_pack(x)
    q_r, s_r, z_r = asym_quantize_and_pack_i4(x)
    check(f"kv {b}x{t}x{h}x{d} q", q_f, q_r)
    check(f"kv {b}x{t}x{h}x{d} scale", param[..., 0:1], s_r)
    check(f"kv {b}x{t}x{h}x{d} zero", param[..., 1:2], z_r)

# a constant row makes xmax == xmin, exercising the clamp(min=1e-5) branch
x = torch.full((1, 4, 32, 128), 1.5, device=DEV, dtype=torch.float16)
q_f, param = kernel.kv_quant_pack(x)
q_r, s_r, z_r = asym_quantize_and_pack_i4(x)
check("kv constant row q", q_f, q_r)
check("kv constant row scale", param[..., 0:1], s_r)

print("\n=== RMSNorm ===")
ulp = 2 ** -10  # fp16 relative spacing; outputs are O(1) after normalization
for rows, cols in ((1, 4096), (16, 4096), (2048, 4096), (1, 11008)):
    x = torch.randn(1, rows, cols, device=DEV, dtype=torch.float16)
    n = signrot.nn.RMSNorm(cols).cuda()
    got, ref = n(x), n.forward_reference(x)
    exact = torch.equal(got, ref)
    check(f"rmsnorm rows={rows} cols={cols}" + (" (bit-exact)" if exact else ""),
          got, ref, exact=False, atol=ref.abs().max().item() * ulp)

print("\n=== rotary embedding (rope_qk vs apply_rotary_pos_emb) ===")
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
for b, t, hq, hk, d in ((1, 1, 32, 32, 128), (16, 1, 32, 32, 128),
                        (1, 2048, 32, 32, 128), (4, 3, 32, 8, 128)):
    q = torch.randn(b, t, hq, d, device=DEV, dtype=torch.float16)
    k = torch.randn(b, t, hk, d, device=DEV, dtype=torch.float16)
    for cos_b in ({1, b} if b > 1 else {1}):
        cos = torch.randn(cos_b, t, d, device=DEV, dtype=torch.float16)
        sin = torch.randn(cos_b, t, d, device=DEV, dtype=torch.float16)
        q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        q_got, k_got = q.clone(), k.clone()
        kernel.rope_qk(q_got, k_got, cos, sin)
        check(f"rope {b}x{t}x{hq}/{hk}x{d} cos_b={cos_b} q", q_got, q_ref)
        check(f"rope {b}x{t}x{hq}/{hk}x{d} cos_b={cos_b} k", k_got, k_ref)

print("\n=== SwiGLU (swiglu vs SiLU()(gate) * up) ===")
silu = torch.nn.SiLU()
for shape in ((1, 1, 11008), (16, 1, 11008), (1, 2048, 11008)):
    gate = torch.randn(*shape, device=DEV, dtype=torch.float16) * 4
    up = torch.randn(*shape, device=DEV, dtype=torch.float16) * 4
    check(f"swiglu {shape}", kernel.swiglu(gate, up), silu(gate) * up)

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} checks")
    for f in FAILED:
        print("  -", f)
    raise SystemExit(1)
print("all equivalence checks passed")
