"""Single linear layer latency, FP16 / Int4 / +QuaRot / +SignRot (table speedup_single_linear)."""

import argparse
import csv
import os
import time

import numpy as np
import torch
from signrot.nn import Linear4bit, OnlineHadamard, Quantizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
# (hidden, intermediate) of LLaMA2-7B/13B/70B; labelled out x in, measured with in = hidden
SIZES = [(4096, 11008), (5120, 13824), (8192, 28672)]
CONFIGS = ["fp16", "int4", "quarot", "signrot"]
LABELS = {"fp16": "FP16", "int4": "Int4", "quarot": "+QuaRot", "signrot": "+SignRot"}


def build(config, fin, fout):
    lin = torch.nn.Linear(fin, fout, bias=False).cuda().half()
    lin.weight.data = torch.randint_like(lin.weight.data, low=-8, high=7)
    if config == "fp16":
        return lin
    scales = torch.ones((fout, 1), dtype=torch.float16, device="cuda")
    layers = [Quantizer(clip_ratio=1.0), Linear4bit.from_float(lin, weight_scales=scales)]
    if config in ("quarot", "signrot"):
        had = OnlineHadamard(fin, force_fp32=False, had_adj=config == "signrot")
        if had.had_adj_sign is not None:
            had.had_adj_sign = (torch.randint(0, 2, (fin,)) * 2 - 1).to(torch.int8)
        layers.insert(0, had)
    return torch.nn.Sequential(*layers).cuda()


def time_ms(module, x, warmup, iters):
    for _ in range(warmup):
        module(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        module(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(REPO, "fake_quant/experiments/reproduce/efficiency"))
    p.add_argument("--bsz", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--down_proj_input", action="store_true", help="input = intermediate size (real down_proj)")
    args = p.parse_args()

    rows = []
    for hidden, inter in SIZES:
        fin, fout = (inter, hidden) if args.down_proj_input else (hidden, inter)
        modules = {c: build(c, fin, fout) for c in CONFIGS}
        for bsz in args.bsz:
            x = torch.rand(bsz, args.seq_len, fin, device="cuda").half()
            row = dict(size=f"{fout}x{fin}", in_features=fin, out_features=fout, bsz=bsz)
            for c in ["int4", "quarot", "signrot", "fp16"]:
                t = [time_ms(modules[c], x, args.warmup, args.iters) for _ in range(args.repeats)]
                row[c], row[f"{c}_std"] = np.mean(t), np.std(t)
            print(row, flush=True)
            rows.append(row)
            del x
            torch.cuda.empty_cache()
        del modules
        torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    name = "speedup_single_linear" + ("_down_proj_input" if args.down_proj_input else "")
    with open(os.path.join(args.out_dir, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"# Single linear layer latency (ms), seq {args.seq_len}, {torch.cuda.get_device_name()}", "",
             "| Layer size | Batch | " + " | ".join(LABELS[c] for c in CONFIGS)
             + " | Speedup | SignRot overhead |",
             "|---|---|" + "---|" * (len(CONFIGS) + 2)]
    for r in rows:
        lines.append(f"| {r['size']} | {r['bsz']} | " + " | ".join(f"{r[c]:.2f}" for c in CONFIGS)
                     + f" | {r['fp16'] / r['signrot']:.2f}x | {100 * (r['signrot'] / r['quarot'] - 1):+.2f}% |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, f"{name}.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
