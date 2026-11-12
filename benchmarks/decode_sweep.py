"""Batch sweep of the --use_kernel decode path on a single GPU.

Builds a synthetic model with the deployment modules (signrot.nn.Linear4bit +
Quantizer + OnlineHadamard + RMSNorm, kernel_utils paged int4 KV cache) so the
graph is identical to `main.py --use_kernel`.  Weights are random: decode
runtime does not depend on weight values, which lets the sweep run without a
checkpoint and without a GPTQ/RTN pass.

Reports, per batch size: wall ms/token, total tok/s, GPU-kernel time from the
profiler (how much of the wall is real GPU work), CUDA launches per token, and
peak memory.  `--fp16` runs the same shapes through plain HF eager fp16 as a
same-framework reference.

Usage:
  .venv/bin/python benchmarks/decode_sweep.py --batches 1 2 4 8 16 32
  .venv/bin/python benchmarks/decode_sweep.py --fp16 --batches 1 8
"""

import argparse
import gc
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake_quant"))

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import kernel_utils
import signrot.nn

# Llama-2-7B
H, I, NL, NH, HD, VOCAB = 4096, 11008, 32, 32, 128, 32000


def build_config(attn_impl):
    return LlamaConfig(
        hidden_size=H, intermediate_size=I, num_hidden_layers=NL,
        num_attention_heads=NH, num_key_value_heads=NH, vocab_size=VOCAB,
        max_position_embeddings=4096, _attn_implementation=attn_impl,
    )


def _init_materialized(model):
    """`to_empty` leaves uninitialised memory; fill it so activations stay finite
    and the model's output actually depends on its input (needed for the
    graph-vs-eager checks, irrelevant to the timings)."""
    for p in model.parameters():
        if p.numel():
            p.data.normal_(0, 0.02)


class KernelLinear(torch.nn.Module):
    """Mirror of quant_utils.ActQuantWrapper._forward_kernel."""

    def __init__(self, in_f, out_f, had=None, had_dim=0, asym=False):
        super().__init__()
        self.lin = signrot.nn.Linear4bit(in_f, out_f).cuda()
        self.lin.weight.data = torch.randint(
            0, 255, (out_f, in_f // 2), dtype=torch.uint8, device="cuda")
        # Runtime does not depend on the values, but the scale does decide
        # whether the residual stream stays inside fp16 across 32 layers, and the
        # output checks need finite logits.  An int4 GEMM over `in_f` random
        # codes accumulates to ~sqrt(in_f) * 16, so 1/in_f keeps each layer's
        # gain below one.
        self.lin.weight_scales.data = torch.full(
            (out_f, 1), 1.0 / in_f, dtype=torch.float16, device="cuda")
        self.q = signrot.nn.AsymQuantizer() if asym else signrot.nn.Quantizer()
        self.had_dim = had_dim
        self.partial = had == "partial"
        self.had = None
        if had == "full":
            self.had = signrot.nn.OnlineHadamard(in_f).cuda()
        elif had == "partial":
            self.had = signrot.nn.OnlineHadamard(had_dim).cuda()

    # mirrors quant_utils.ActQuantWrapper's kernel path, including the
    # quantize/matmul split that lets q/k/v and gate/up share one quantization
    def kernel_quantize(self, x):
        x = x.to(torch.float16)
        if self.had is not None:
            if self.partial:
                shape = x.shape
                x = x.reshape(-1, shape[-1] // self.had_dim, self.had_dim).transpose(1, 2)
                x = self.had(x)
                x = x.transpose(1, 2).reshape(shape)
            else:
                x = self.had(x)
        return self.q(x)

    def kernel_matmul(self, packed_x, x_dtype):
        return self.lin(packed_x).to(x_dtype)

    def shares_kernel_input_with(self, other):
        return (isinstance(other, KernelLinear) and self.had is None
                and other.had is None and type(self.q) is type(other.q)
                and self.lin.in_features == other.lin.in_features)

    def forward(self, x):
        return self.kernel_matmul(self.kernel_quantize(x), x.dtype)


def build_kernel_model(asym=False):
    torch.set_default_dtype(torch.float16)
    with torch.device("meta"):
        model = LlamaForCausalLM(build_config("flash_attention_2"))
    # the linears are replaced below; do not materialise their fp16 storage
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        a.q_proj = a.k_proj = a.v_proj = a.o_proj = None
        m.gate_proj = m.up_proj = m.down_proj = None
    model = model.to_empty(device="cuda")
    _init_materialized(model)

    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        a.q_proj = KernelLinear(H, H, asym=asym)
        a.k_proj = KernelLinear(H, H, asym=asym)
        a.v_proj = KernelLinear(H, H, asym=asym)
        a.o_proj = KernelLinear(H, H, had="partial", had_dim=HD, asym=asym)
        m.gate_proj = KernelLinear(H, I, asym=asym)
        m.up_proj = KernelLinear(H, I, asym=asym)
        m.down_proj = KernelLinear(I, H, had="full", asym=asym)
        layer.input_layernorm = signrot.nn.RMSNorm(H).cuda()
        layer.post_attention_layernorm = signrot.nn.RMSNorm(H).cuda()
    model.model.norm = signrot.nn.RMSNorm(H).cuda()
    kernel_utils.patch_kernel_model(model, types.SimpleNamespace(k_bits=4))
    return model.eval()


def build_fp16_model():
    torch.set_default_dtype(torch.float16)
    with torch.device("meta"):
        model = LlamaForCausalLM(build_config("sdpa"))
    model = model.to_empty(device="cuda")
    _init_materialized(model)
    return model.eval()


@torch.no_grad()
def run_one(model, bsz, prefill, steps, profile=False, graph=False):
    dev = "cuda"
    ids = torch.randint(1000, 8000, (bsz, prefill), device=dev)
    nxt = torch.randint(1000, 8000, (bsz, 1), device=dev)

    if hasattr(model, "_expected_max_length"):
        model._expected_max_length = prefill + steps
    cache = model(ids, use_cache=True, logits_to_keep=1).past_key_values

    decoder = None
    if graph:
        decoder = kernel_utils.try_graph_decoder(model, cache, bsz)
        if decoder is None:
            raise RuntimeError("CUDA graph capture failed")
    step = ((lambda: decoder.step(nxt)) if decoder is not None
            else (lambda: model(nxt, past_key_values=cache, use_cache=True)))

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    cache.crop(prefill)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1000

    gpu_ms, launches = float("nan"), float("nan")
    if profile:
        from torch.profiler import ProfilerActivity, profile as tprof
        n = 5
        cache.crop(prefill)
        with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(n):
                step()
            torch.cuda.synchronize()
        evs = p.key_averages()
        gpu_ms = sum(e.self_device_time_total for e in evs) / 1e3 / n
        launches = sum(e.count for e in evs
                       if e.key in ("cudaLaunchKernel", "cudaGraphLaunch")) / n

    peak = torch.cuda.max_memory_allocated() / 2**30
    del cache, ids, nxt, decoder, step
    gc.collect()
    torch.cuda.empty_cache()
    return ms, gpu_ms, launches, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="HF eager fp16 reference")
    ap.add_argument("--asym", action="store_true", help="asymmetric activation quantizer")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--graph", action="store_true", help="replay decode from a CUDA graph")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    model = build_fp16_model() if args.fp16 else build_kernel_model(asym=args.asym)
    label = args.label or ("fp16 HF eager" if args.fp16 else "int4 kernel path")
    if args.graph:
        label += " + CUDA graph"
    weights_gb = torch.cuda.memory_allocated() / 2**30
    print(f"\n=== {label} | Llama-2-7B | prefill={args.prefill} steps={args.steps} "
          f"| weights {weights_gb:.2f} GB ===")
    print(f"{'bsz':>4} {'ms/token':>9} {'tok/s':>9} {'tok/s/seq':>10} "
          f"{'gpu ms':>8} {'gpu%':>6} {'launches':>9} {'peak GB':>8}")

    for bsz in args.batches:
        torch.cuda.reset_peak_memory_stats()
        try:
            ms, gpu_ms, launches, peak = run_one(
                model, bsz, args.prefill, args.steps,
                profile=args.profile, graph=args.graph)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{bsz:>4}  OOM (stopping sweep)")
            break
        print(f"{bsz:>4} {ms:9.2f} {bsz / ms * 1000:9.1f} {1000 / ms:10.1f} "
              f"{gpu_ms:8.2f} {gpu_ms / ms * 100:5.0f}% {launches:9.0f} {peak:8.2f}")


if __name__ == "__main__":
    main()
