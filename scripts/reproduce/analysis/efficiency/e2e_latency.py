"""Aggregate efficiency_e2e_latency.sh runs into the end-to-end latency tables (e2e_latency, e2e_latency_full)."""

import argparse
import ast
import glob
import os
import re

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
RESULTS_RE = re.compile(r"Runtime benchmark results:\s*(\[.*\])\s*$", re.M)
OOM_RE = re.compile(r"OutOfMemoryError|out of memory|does not fit")
METRICS = {"prefill_ms": "Prefill (ms)", "decode_ms_per_token": "Decode (ms/token)",
           "e2e_ms": "E2E (ms)", "e2e_peak_mem_gb": "Peak memory (GB)"}
CONFIGS = {"fp16": ("FP16", "-"), "quarot_sym": ("QuaRot", "sym"), "signrot_sym": ("QuaRot+SignRot", "sym"),
           "quarot_asym": ("QuaRot", "asym"), "signrot_asym": ("QuaRot+SignRot", "asym")}


def collect(exp_root, run_dir):
    rows = []
    for log in sorted(glob.glob(os.path.join(exp_root, "*", "*", run_dir, "*", "r*", "*", "rank0.log"))):
        parts = os.path.relpath(log, exp_root).split(os.sep)
        model, (cfg, rnd) = "/".join(parts[:2]), parts[-4:-2]
        text = open(log, errors="replace").read()
        found = RESULTS_RE.findall(text)
        base = dict(model=model, config=cfg, round=rnd, log=log)
        if not found:
            stdout = os.path.join(os.path.dirname(os.path.dirname(log)), "stdout.txt")
            if os.path.exists(stdout):
                text += open(stdout, errors="replace").read()
            rows.append({**base, "status": "OOM" if OOM_RE.search(text) else "failed"})
            continue
        for res in ast.literal_eval(re.sub(r"np\.(?:float|int)\d*\(", "(", found[-1])):
            rows.append({**base, "bsz": res["bsz"], **{k: res[k] for k in METRICS}})
    return pd.DataFrame(rows, columns=["model", "config", "round", "log", "status", "bsz", *METRICS])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_root", default=os.path.join(REPO, "fake_quant/experiments"))
    p.add_argument("--run_dir", default="reproduce/efficiency/e2e")
    p.add_argument("--out_dir", default=os.path.join(REPO, "fake_quant/experiments/reproduce/efficiency"))
    args = p.parse_args()

    df = collect(args.exp_root, args.run_dir)
    if df.empty:
        raise SystemExit(f"no runs under {args.exp_root}/*/*/{args.run_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "e2e_runs.csv"), index=False)

    text = "# End-to-end latency (bsz 1; mean over rounds)\n"
    for model, mdf in df.groupby("model"):
        mean = mdf.dropna(subset=["e2e_ms"]).groupby("config")[list(METRICS)].mean()
        n = mdf.dropna(subset=["e2e_ms"]).groupby("config")["round"].nunique()
        lines = [f"\n## {model}\n", "| Method | Act quant | Rounds | " + " | ".join(
            f"{v} | {'Saving' if k == 'e2e_peak_mem_gb' else 'Speedup'}" for k, v in METRICS.items()) + " |",
                 "|---|---|---|" + "---|---|" * len(METRICS)]
        for cfg, (name, act) in CONFIGS.items():
            if cfg not in set(mdf.config):
                continue
            if cfg not in mean.index:
                status = "/".join(sorted(set(mdf[mdf.config == cfg].status.dropna())))
                lines.append(f"| {name} | {act} | - | " + " | ".join([status, ""] * len(METRICS)) + " |")
                continue
            cells = []
            for k in METRICS:
                v = mean.loc[cfg, k]
                ratio = f"{mean.loc['fp16', k] / v:.2f}x" if "fp16" in mean.index else "-"
                cells += [f"{v:.2f}", ratio]
            lines.append(f"| {name} | {act} | {n[cfg]} | " + " | ".join(cells) + " |")
        for act in ("sym", "asym"):
            q, s = f"quarot_{act}", f"signrot_{act}"
            if q in mean.index and s in mean.index:
                ov = [f"{100 * (mean.loc[s, k] / mean.loc[q, k] - 1):+.2f}%" for k in METRICS]
                lines.append(f"| Overhead of SignRot | {act} | | " + " | ".join(f"{o} | " for o in ov) + " |")
        text += "\n".join(lines) + "\n"

    with open(os.path.join(args.out_dir, "e2e_latency.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
