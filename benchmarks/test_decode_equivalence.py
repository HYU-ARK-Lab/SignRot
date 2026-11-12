"""End-to-end check that the optimized decode path did not change the model.

Runs a full Llama-2-7B-shaped kernel-path model twice from the same cache state:
once as shipped (fused quantizers/RMSNorm/RoPE/SwiGLU, split-KV attention,
shared q/k/v quantization, CUDA graph) and once with every one of those swapped
back for the implementation it replaced, then compares prefill and decode logits.

Run: .venv/bin/python benchmarks/test_decode_equivalence.py
"""

import math
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake_quant"))

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import kernel_utils
import signrot.functional.kernel as K
import signrot.nn
from decode_sweep import build_kernel_model
from signrot.transformers.kv_cache import asym_quantize_and_pack_i4

DEV = "cuda"
B, PRE, STEPS = 2, 512, 8


class ReferenceImpls:
    """Swap every fused kernel back for the code path it replaced."""

    def __enter__(self):
        self.saved = {
            "sym": signrot.nn.Quantizer.forward,
            "asym": signrot.nn.AsymQuantizer.forward,
            "norm": signrot.nn.RMSNorm.forward,
            "rope": K.rope_qk,
            "swiglu": K.swiglu,
            "kv": K.kv_quant_pack,
            "shares": kernel_utils._shares_input,
        }
        signrot.nn.Quantizer.forward = signrot.nn.Quantizer.forward_reference
        signrot.nn.AsymQuantizer.forward = signrot.nn.AsymQuantizer.forward_reference
        signrot.nn.RMSNorm.forward = signrot.nn.RMSNorm.forward_reference

        def rope_ref(q, k, cos, sin):
            qe, ke = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
            q.copy_(qe)
            k.copy_(ke)
            return q, k

        K.rope_qk = rope_ref
        K.swiglu = lambda gate, up: torch.nn.functional.silu(gate) * up
        K.kv_quant_pack = lambda x: (
            lambda q, s, z: (q, torch.cat([s, z], dim=-1))
        )(*asym_quantize_and_pack_i4(x))
        kernel_utils._shares_input = lambda *a: False
        return self

    def __exit__(self, *exc):
        signrot.nn.Quantizer.forward = self.saved["sym"]
        signrot.nn.AsymQuantizer.forward = self.saved["asym"]
        signrot.nn.RMSNorm.forward = self.saved["norm"]
        K.rope_qk = self.saved["rope"]
        K.swiglu = self.saved["swiglu"]
        K.kv_quant_pack = self.saved["kv"]
        kernel_utils._shares_input = self.saved["shares"]


def relative(a, b):
    denom = max(b.abs().max().item(), 1e-6)
    return (a.float() - b.float()).abs().max().item() / denom


FAILED = []


def check(name, got, ref, tol):
    for label, t in (("got", got), ("ref", ref)):
        if not torch.isfinite(t).all():
            raise AssertionError(
                f"{name}: {label} logits are not finite -- the synthetic weights "
                f"overflowed, so the comparison would be meaningless")
    rel = relative(got, ref)
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    ok = rel <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:38s} rel {rel:.3e} (tol {tol:.0e})"
          f"  argmax agreement {agree * 100:.1f}%")
    if not ok:
        FAILED.append(name)


@torch.no_grad()
def main():
    model = build_kernel_model()
    ids = torch.randint(1000, 8000, (B, PRE), device=DEV)
    toks = [torch.randint(1000, 8000, (B, 1), device=DEV) for _ in range(STEPS)]

    def run(eager_decode):
        model._expected_max_length = PRE + STEPS + 8
        out = model(ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        prefill = out.logits.clone()
        if eager_decode:
            steps = [model(t, past_key_values=cache, use_cache=True).logits.clone()
                     for t in toks]
        else:
            dec = kernel_utils.try_graph_decoder(model, cache, B)
            assert dec is not None, "CUDA graph capture failed"
            steps = [dec.step(t).clone() for t in toks]
        del cache
        torch.cuda.empty_cache()
        return prefill, steps

    print("=== optimized (graph) vs optimized (eager) ===")
    opt_prefill, opt_graph = run(eager_decode=False)
    _, opt_eager = run(eager_decode=True)
    for i, (g, e) in enumerate(zip(opt_graph, opt_eager)):
        if i in (0, STEPS - 1):
            check(f"decode step {i} graph vs eager", g, e, 0.0)

    print("\n=== optimized vs reference implementations ===")
    with ReferenceImpls():
        ref_prefill, ref_steps = run(eager_decode=True)
    check("prefill logits", opt_prefill, ref_prefill, 2e-3)
    for i, (o, r) in enumerate(zip(opt_graph, ref_steps)):
        if i in (0, STEPS // 2, STEPS - 1):
            check(f"decode step {i} logits", o, r, 5e-3)

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} checks")
        for f in FAILED:
            print("  -", f)
        raise SystemExit(1)
    print("optimized decode path matches the reference implementations")


if __name__ == "__main__":
    main()
