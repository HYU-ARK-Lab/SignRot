"""The kernel-work floor for one Llama-2-7B decode token.

Times just the 224 int4 GEMMs a decode step performs (7 per layer x 32 layers)
against distinct weights, inside a CUDA graph so no launch overhead is counted.
That is the "CUTLASS int4 GEMM only" target the full decode path is measured
against.  Reading the int4 KV cache is unavoidable too and scales with the
batch, so the last column adds its bandwidth-limited cost: at batch 1 the GEMMs
dominate, by batch 16+ the KV read is comparable.
"""

import argparse

import torch

import signrot.functional.kernel as K

NL, HEADS, HEAD_DIM = 32, 32, 128
# (in_features, out_features, count per layer)
SHAPES = [(4096, 4096, 4), (4096, 11008, 2), (11008, 4096, 1)]


def graph_time(fn, inner=20, outer=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(outer):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / (inner * outer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--ctx", type=int, default=2048,
                    help="context length used for the KV-read term")
    args = ap.parse_args()

    # One distinct weight per GEMM, as in the real model: all 3.2 GB of int4
    # weights are streamed once per token, so nothing stays resident in L2.
    weights, inputs = [], {}
    for in_f, out_f, cnt in SHAPES:
        for _ in range(cnt * NL):
            weights.append((in_f, torch.randint(
                0, 255, (out_f, in_f // 2), dtype=torch.uint8, device="cuda")))
    total_gb = sum(w.numel() for _, w in weights) / 2**30
    print(f"int4 weights streamed per token: {total_gb:.2f} GB "
          f"({len(weights)} GEMMs / token)\n")

    # Reading the int4 KV cache is unavoidable work too, and it grows with the
    # batch, so at larger batches the GEMM-only number stops being the floor.
    bw = measure_bandwidth()
    print(f"measured device copy bandwidth: {bw:.0f} GB/s\n")

    print(f"{'bsz':>4} {'gemm ms':>9} {'tok/s':>9} {'GB/s':>7} "
          f"{'kv GB/tok':>10} {'kv ms':>7} {'gemm+kv ms':>11}")
    for bsz in args.batches:
        for in_f, _, _ in SHAPES:
            inputs[in_f] = torch.randint(
                0, 255, (bsz, in_f // 2), dtype=torch.uint8, device="cuda")

        def one_token():
            for in_f, w in weights:
                K.matmul_si4_si4(inputs[in_f], w)

        ms = graph_time(one_token, inner=2, outer=10)
        # K and V, int4, every layer, for the whole context
        kv_gb = bsz * args.ctx * HEADS * HEAD_DIM * 0.5 * 2 * NL / 1e9
        kv_ms = kv_gb / bw * 1000
        print(f"{bsz:>4} {ms:9.2f} {bsz / ms * 1000:9.1f} "
              f"{total_gb * 2**30 / (ms / 1000) / 1e9:7.0f} "
              f"{kv_gb:10.2f} {kv_ms:7.2f} {ms + kv_ms:11.2f}")


def measure_bandwidth():
    big = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(big)
    ms = graph_time(lambda: dst.copy_(big), inner=2, outer=5)
    gb = 2 * big.numel() / 1e9
    del big, dst
    torch.cuda.empty_cache()
    return gb / (ms / 1000)


if __name__ == "__main__":
    main()
