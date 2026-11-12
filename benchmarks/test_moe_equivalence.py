"""SignRotMoEBlock vs the HF per-expert loop.

Builds a Qwen3MoeSparseMoeBlock whose expert projections mirror the
ActQuantWrapper kernel path (Linear4bit + Quantizer/AsymQuantizer +
OnlineHadamard with a per-expert had_adj sign), then checks that
SignRotMoEBlock over the stacked weights reproduces it: router logits
bit-exact, block output within fp16 accumulation-order tolerance (the HF loop
accumulates with index_add_, the stacked block with a sorted sum -- everything
up to the down-proj GEMM is the same kernels on the same rows).

Run: .venv/bin/python benchmarks/test_moe_equivalence.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake_quant"))

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

import signrot.nn
from signrot.nn.moe import SignRotMoEBlock

DEV = "cuda"
REL_TOL = 2e-3

failures = []


class ExpertLinear(torch.nn.Module):
    """Mirror of quant_utils.ActQuantWrapper's kernel path for one projection."""

    def __init__(self, in_f, out_f, had=False, sign=None, asym=False, clip_ratio=1.0,
                 fp32_had=False):
        super().__init__()
        self.lin = signrot.nn.Linear4bit(in_f, out_f).to(DEV)
        self.lin.weight.data = torch.randint(
            0, 256, (out_f, in_f // 2), dtype=torch.uint8, device=DEV)
        # keep each stage's gain around one so activations stay finite in fp16
        self.lin.weight_scales.data = torch.full(
            (out_f, 1), 1.0 / in_f, dtype=torch.float16, device=DEV)
        self.q = (signrot.nn.AsymQuantizer(clip_ratio=clip_ratio) if asym
                  else signrot.nn.Quantizer(clip_ratio=clip_ratio))
        self.had = None
        if had:
            self.had = signrot.nn.OnlineHadamard(
                in_f, force_fp32=fp32_had, had_adj=sign is not None).to(DEV)
            if sign is not None:
                self.had.had_adj_sign.copy_(sign)

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float16)
        if self.had is not None:
            x = self.had(x)
        return self.lin(self.q(x)).to(x_dtype)


def build_oracle(cfg, asym, sign_tables, clip_ratio=1.0, fp32_had=False,
                 dtype=torch.float16):
    with torch.device("meta"):
        block = Qwen3MoeSparseMoeBlock(cfg)
    block.gate = torch.nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False,
                                 dtype=dtype, device=DEV)
    torch.nn.init.normal_(block.gate.weight, std=0.1)
    h, i = cfg.hidden_size, cfg.moe_intermediate_size
    for e, expert in enumerate(block.experts):
        expert.gate_proj = ExpertLinear(h, i, asym=asym, clip_ratio=clip_ratio)
        expert.up_proj = ExpertLinear(h, i, asym=asym, clip_ratio=clip_ratio)
        expert.down_proj = ExpertLinear(
            i, h, had=True, sign=None if sign_tables is None else sign_tables[e],
            asym=asym, clip_ratio=clip_ratio, fp32_had=fp32_had)
    return block.eval()


def build_stacked(block, cfg, asym, sign_tables, clip_ratio=1.0, fp32_had=False):
    w_gate, s_gate, wsum_gate = SignRotMoEBlock.stack_experts(
        [e.gate_proj.lin for e in block.experts], asym)
    w_up, s_up, wsum_up = SignRotMoEBlock.stack_experts(
        [e.up_proj.lin for e in block.experts], asym)
    w_down, s_down, wsum_down = SignRotMoEBlock.stack_experts(
        [e.down_proj.lin for e in block.experts], asym)
    return SignRotMoEBlock(
        block.gate, top_k=cfg.num_experts_per_tok, norm_topk_prob=cfg.norm_topk_prob,
        w_gate=w_gate, s_gate=s_gate, w_up=w_up, s_up=s_up,
        w_down=w_down, s_down=s_down, sign_down=sign_tables,
        wsum_gate=wsum_gate, wsum_up=wsum_up, wsum_down=wsum_down,
        asym=asym, clip_ratio=clip_ratio, fp32_had=fp32_had,
    ).eval()


def report(name, ok, detail):
    global failures
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<52} {detail}")
    if not ok:
        failures.append(name)


@torch.no_grad()
def check(name, oracle, stacked, shape, dtype=torch.float16):
    x = torch.randn(*shape, oracle.gate.in_features, device=DEV, dtype=dtype)
    ref, ref_logits = oracle(x)
    stacked.use_grouped = True
    out, logits = stacked(x)
    stacked.use_grouped = False
    out_loop, _ = stacked(x)
    stacked.use_grouped = True
    if not torch.equal(ref_logits, logits):
        report(name, False, "router logits differ")
        return
    # grouped GEMMs/quant/dequant are bit-identical to the per-expert loop
    if not torch.equal(out, out_loop):
        rel_gl = ((out - out_loop).abs().max() / out.abs().max().clamp_min(1e-6)).item()
        report(name, False, f"grouped != loop (rel {rel_gl:.2e})")
        return
    rel = ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    # the HF loop rounds up to top_k sequential index_add_ steps in the model
    # dtype; in bf16 (eps 2^-8) that alone is ~top_k * eps
    tol = REL_TOL if dtype == torch.float16 else 4e-2
    report(name, rel < tol, f"rel {rel:.2e}")


@torch.no_grad()
def check_capture(name, stacked, shape):
    """Capture one grouped forward in a CUDA graph, then replay it on inputs
    the capture never saw -- routing is data-dependent, so this only passes if
    the whole pipeline is fixed-shape and sync-free."""
    hidden = stacked.hidden_size
    x_static = torch.randn(*shape, hidden, device=DEV, dtype=torch.float16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            stacked(x_static)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out_static, _ = stacked(x_static)
    except Exception as exc:
        report(name, False, f"capture failed: {exc}")
        return

    worst = 0.0
    for _ in range(3):
        x_new = torch.randn_like(x_static)
        x_static.copy_(x_new)
        graph.replay()
        replay = out_static.clone()
        ref, _ = stacked(x_new)
        rel = ((replay - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        worst = max(worst, rel)
    report(name, worst < 1e-3, f"replay vs eager rel {worst:.2e}")


def make_signs(cfg, uncalibrated_tail=8):
    """Random +-1 per expert; the last few all-ones like never-routed experts."""
    signs = torch.randint(0, 2, (cfg.num_experts, cfg.moe_intermediate_size),
                          dtype=torch.int8, device=DEV) * 2 - 1
    signs[-uncalibrated_tail:] = 1
    return signs


@torch.no_grad()
def main():
    torch.manual_seed(0)

    # Qwen3-30B-A3B geometry
    cfg = Qwen3MoeConfig(hidden_size=2048, moe_intermediate_size=768,
                         num_experts=128, num_experts_per_tok=8, norm_topk_prob=True)
    signs = make_signs(cfg)

    print("=== sym activations, sign-adjusted down_proj ===")
    oracle = build_oracle(cfg, asym=False, sign_tables=signs)
    stacked = build_stacked(oracle, cfg, asym=False, sign_tables=signs)
    for shape in [(1, 1), (1, 8), (1, 33), (4, 7), (1, 2048)]:
        check(f"sym sign {shape}", oracle, stacked, shape)
    oracle.gate.to(torch.bfloat16)  # shared with `stacked`; bf16 model = bf16 router
    check("sym sign (2, 5) bf16 input", oracle, stacked, (2, 5), dtype=torch.bfloat16)
    oracle.gate.to(torch.float16)

    print("=== sym, skewed routing (a few experts take most tokens) ===")
    oracle.gate.weight.data[:4] += 2.0
    for shape in [(1, 64), (1, 512)]:
        check(f"sym skew {shape}", oracle, stacked, shape)
    oracle.gate.weight.data[:4] -= 2.0

    print("=== CUDA graph capture (grouped path, unseen inputs) ===")
    check_capture("sym capture decode (1, 1)", stacked, (1, 1))
    check_capture("sym capture batch (8, 1)", stacked, (8, 1))
    del oracle, stacked
    torch.cuda.empty_cache()

    print("=== asym activations ===")
    oracle = build_oracle(cfg, asym=True, sign_tables=signs)
    stacked = build_stacked(oracle, cfg, asym=True, sign_tables=signs)
    for shape in [(1, 1), (1, 33), (4, 7), (1, 2048)]:
        check(f"asym sign {shape}", oracle, stacked, shape)
    check_capture("asym capture decode (1, 1)", stacked, (1, 1))
    del oracle, stacked
    torch.cuda.empty_cache()

    print("=== no sign table (quarot preset), norm_topk_prob off ===")
    cfg_nonorm = Qwen3MoeConfig(hidden_size=2048, moe_intermediate_size=768,
                                num_experts=128, num_experts_per_tok=8,
                                norm_topk_prob=False)
    oracle = build_oracle(cfg_nonorm, asym=False, sign_tables=None)
    stacked = build_stacked(oracle, cfg_nonorm, asym=False, sign_tables=None)
    check("sym no-sign no-norm (1, 17)", oracle, stacked, (1, 17))
    del oracle, stacked
    torch.cuda.empty_cache()

    print("=== fp32 hadamard / clip_ratio 0.9 ===")
    oracle = build_oracle(cfg, asym=False, sign_tables=signs, fp32_had=True)
    stacked = build_stacked(oracle, cfg, asym=False, sign_tables=signs, fp32_had=True)
    check("sym sign fp32_had (1, 9)", oracle, stacked, (1, 9))
    del oracle, stacked
    torch.cuda.empty_cache()
    oracle = build_oracle(cfg, asym=True, sign_tables=signs, clip_ratio=0.9)
    stacked = build_stacked(oracle, cfg, asym=True, sign_tables=signs, clip_ratio=0.9)
    check("asym sign clip 0.9 (1, 16)", oracle, stacked, (1, 16))
    del oracle, stacked
    torch.cuda.empty_cache()

    print("=== small non-Qwen geometry (E=16 topk=4 H=256 I=192) ===")
    cfg_small = Qwen3MoeConfig(hidden_size=256, moe_intermediate_size=192,
                               num_experts=16, num_experts_per_tok=4,
                               norm_topk_prob=True)
    signs_small = make_signs(cfg_small, uncalibrated_tail=2)
    for asym in (False, True):
        oracle = build_oracle(cfg_small, asym=asym, sign_tables=signs_small)
        stacked = build_stacked(oracle, cfg_small, asym=asym, sign_tables=signs_small)
        check(f"small {'asym' if asym else 'sym'} (3, 11)", oracle, stacked, (3, 11))
        del oracle, stacked

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("\nall MoE equivalence checks passed")


if __name__ == "__main__":
    main()
