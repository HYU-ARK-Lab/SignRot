"""Tables ma_statistics / ma_statistics_large: per-layer massive outlier statistics (Sun et al. vs ours)."""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

ABS_TH, REL_TH = 100.0, 1000.0
MASS_FRAC, TRUE_MA_PEAK = 0.5, 100.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help=".pt from capture_topk.py")
    p.add_argument("--out", required=True, help="output csv (a .md sibling is written too)")
    return p.parse_args()


def site_tokens(entry):
    """Per-token Sun channel count, recursive depth (0 = not ours), M_1, M_2 and ||x||^2 of one site."""
    v = entry["topk_values"].double()
    total = entry["sq_sum"].double()
    med = entry["median_abs"].double()
    cum = (v**2).cumsum(1)
    tail = cum[:, -1:] - cum
    rest = torch.maximum(total[:, None] - cum, tail).clamp_min(1e-12)
    sun = ((v > ABS_TH) & (v > REL_TH * med[:, None])).sum(1)
    run = torch.cumprod((v**2 / rest > 1).int(), dim=1).sum(1)
    peak = float(v[:, 0].max())
    gate = (v[:, 0] >= MASS_FRAC * peak) if peak >= TRUE_MA_PEAK else torch.zeros_like(run, dtype=torch.bool)
    depth = torch.where(gate, run, 0)
    return sun.numpy(), depth.numpy(), v[:, 0].numpy(), v[:, 1].numpy(), total.numpy()


def model_rows(path):
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = data["meta"]
    per_layer = defaultdict(list)
    for entry in data["layers"].values():
        if entry.get("num_tokens", 0) > 0:
            per_layer[entry["layer_idx"]].append(site_tokens(entry))
    rows = []
    for layer in sorted(per_layer):
        sun, depth, m1, m2, tot = (np.concatenate(x) for x in zip(*per_layer[layer]))
        s, r = sun >= 1, depth >= 1
        if not s.any() and not r.any():
            continue
        rows.append({"model": meta["model"], "layer": layer + 1,
                     "sun_tokens": int(s.sum()), "ours_tokens": int(r.sum()),
                     "sun_tokens_per_seq": s.sum() / meta["nsamples"], "ours_tokens_per_seq": r.sum() / meta["nsamples"],
                     "sun_channels_per_token": sun[s].mean() if s.any() else np.nan,
                     "ours_channels_per_token": depth[r].mean() if r.any() else np.nan,
                     "M2_over_M1_median": float(np.median(m2[r] / m1[r])) if r.any() else np.nan,
                     "M1_energy_pct_median": 100 * float(np.median(m1[r] ** 2 / tot[r])) if r.any() else np.nan})
    return rows


def main():
    args = parse_args()
    rows = []
    for path in args.input:
        r = model_rows(path)
        print(f"{path}: {len(r)} layers with massive outlier tokens")
        rows += r
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    paper = df[["model", "layer", "sun_tokens_per_seq", "ours_tokens_per_seq", "sun_channels_per_token",
                "ours_channels_per_token", "M2_over_M1_median", "M1_energy_pct_median"]]
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("# Massive outlier tokens at the down projection input, per layer (Sun et al. vs ours)\n\n")
        f.write(paper.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(paper.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
