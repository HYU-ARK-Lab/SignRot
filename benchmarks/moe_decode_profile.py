"""Where a Qwen3-MoE decode token actually goes.

Batch-1 decode on a Mixture-of-Experts model is dominated by the router's
host-side dispatch rather than by the int4 GEMMs: HF's Qwen3MoeSparseMoeBlock
reads `expert_hit.nonzero()` back to the CPU and then loops over the selected
experts in Python, which both forces a device->host sync per layer and makes the
step impossible to capture into a CUDA graph.

This builds ONE MoE block with int4 experts (not the whole 48-layer model, which
would need 14GB) and separates:
  * total block time
  * the int4 GEMMs alone, issued back to back without the router
  * the router/dispatch remainder

then scales to a full token by the layer count.

Run: .venv/bin/python benchmarks/moe_decode_profile.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake_quant"))

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from decode_sweep import KernelLinear
from signrot.nn.moe import SignRotMoEBlock

# Qwen3-30B-A3B
LAYERS, HIDDEN, EXPERTS, TOPK, MOE_INTER = 48, 2048, 128, 8, 768
DEV = "cuda"


def wall_ms(fn, n=20, warm=5, trials=5):
    """Minimum over several trials.

    This path is host-bound, so any other load on the machine inflates it; the
    minimum is the measurement least disturbed by interference.
    """
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / n * 1000)
    return best


def build_block():
    cfg = Qwen3MoeConfig(
        hidden_size=HIDDEN, num_experts=EXPERTS, num_experts_per_tok=TOPK,
        moe_intermediate_size=MOE_INTER, num_hidden_layers=LAYERS,
    )
    torch.set_default_dtype(torch.float16)
    block = Qwen3MoeSparseMoeBlock(cfg).to(DEV).half()
    for expert in block.experts:
        expert.gate_proj = KernelLinear(HIDDEN, MOE_INTER)
        expert.up_proj = KernelLinear(HIDDEN, MOE_INTER)
        expert.down_proj = KernelLinear(MOE_INTER, HIDDEN, had="full")
    return block.eval()


@torch.no_grad()
def main():
    block = build_block()
    x = torch.randn(1, 1, HIDDEN, device=DEV, dtype=torch.float16)

    total = wall_ms(lambda: block(x))

    # the same int4 GEMMs the top-k experts run, issued straight through with no
    # router, no gather/scatter and no host sync
    hit = [block.experts[i] for i in range(TOPK)]
    xin = torch.randn(1, HIDDEN, device=DEV, dtype=torch.float16)

    def gemms_only():
        for e in hit:
            e.down_proj(e.gate_proj(xin) * e.up_proj(xin))

    gemm = wall_ms(gemms_only)

    # how much of the block is the router itself (gate + softmax + topk + one_hot
    # + nonzero + the Python loop), measured with the expert bodies stubbed out
    real_forward = [e.forward for e in block.experts]
    zero = torch.zeros(1, HIDDEN, device=DEV, dtype=torch.float16)
    for e in block.experts:
        e.forward = lambda t, _z=zero: _z.expand(t.shape[0], HIDDEN)
    router = wall_ms(lambda: block(x))
    for e, f in zip(block.experts, real_forward):
        e.forward = f

    print(f"Qwen3-30B-A3B MoE block, batch 1, one decode token")
    print(f"  {'full block':<34} {total:8.3f} ms")
    print(f"  {'  int4 GEMMs alone (top-8 experts)':<34} {gemm:8.3f} ms")
    print(f"  {'  router + dispatch (experts stubbed)':<34} {router:8.3f} ms")
    print(f"  {'  unaccounted (gather/scatter etc.)':<34} {total - gemm - router:8.3f} ms")
    print()
    print(f"scaled to {LAYERS} layers (MLP only, excludes attention):")
    print(f"  {'full MoE MLP':<34} {total * LAYERS:8.1f} ms/token")
    print(f"  {'  int4 GEMMs':<34} {gemm * LAYERS:8.1f} ms/token "
          f"({gemm / total * 100:.0f}%)")
    print(f"  {'  router + dispatch':<34} {router * LAYERS:8.1f} ms/token "
          f"({router / total * 100:.0f}%)")

    # is it host-bound?  compare CPU issue time against wall
    for _ in range(5):
        block(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        block(x)
    cpu = (time.perf_counter() - t) / 20 * 1000
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / 20 * 1000
    print(f"\n  cpu issue {cpu:.3f} ms vs wall {wall:.3f} ms "
          f"-> {'host-bound' if cpu > 0.9 * wall else 'GPU-bound'}")

    # the stacked block: GPU-side routing + grouped int4 GEMMs + fused
    # per-expert sign/quant/dequant, one code path for any batch
    signs = torch.randint(0, 2, (EXPERTS, MOE_INTER), dtype=torch.int8, device=DEV) * 2 - 1
    w_gate, s_gate, _ = SignRotMoEBlock.stack_experts(
        [e.gate_proj.lin for e in block.experts], False)
    w_up, s_up, _ = SignRotMoEBlock.stack_experts(
        [e.up_proj.lin for e in block.experts], False)
    w_down, s_down, _ = SignRotMoEBlock.stack_experts(
        [e.down_proj.lin for e in block.experts], False)
    stacked = SignRotMoEBlock(
        block.gate, top_k=TOPK, norm_topk_prob=True,
        w_gate=w_gate, s_gate=s_gate, w_up=w_up, s_up=s_up,
        w_down=w_down, s_down=s_down, sign_down=signs).eval()

    t_stacked = wall_ms(lambda: stacked(x))

    x_static = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            stacked(x_static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stacked(x_static)

    def replay():
        x_static.copy_(x)
        graph.replay()

    t_graph = wall_ms(replay)

    print(f"\nstacked SignRotMoEBlock (grouped GEMM + fused sign/quant):")
    print(f"  {'eager':<34} {t_stacked:8.3f} ms  ({total / t_stacked:4.1f}x vs HF loop)")
    print(f"  {'CUDA graph replay':<34} {t_graph:8.3f} ms  ({total / t_graph:4.1f}x vs HF loop)")
    print(f"  scaled to {LAYERS} layers: eager {t_stacked * LAYERS:7.1f} ms/token, "
          f"graph {t_graph * LAYERS:7.1f} ms/token")


if __name__ == "__main__":
    main()
