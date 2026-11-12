"""Table early_ma_layers: massive outlier layers within the first sixth of each network."""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help=".npz from range_bound.py")
    p.add_argument("--out", required=True, help="output csv (a .md sibling is written too)")
    return p.parse_args()


def model_rows(path):
    z = np.load(path)
    df = pd.DataFrame(z["rec"], columns=[str(c) for c in z["cols"]])
    n_layers, n_seq = int(z["num_layers"]), int(z["nsamples"]) - 1  # sequence 0 only calibrates
    ma = df[df.depth >= 1]
    rows = []
    for li, t in ma.groupby("layer"):
        layer = int(li) + 1
        if layer > n_layers / 6:
            continue
        ok = t[t.range_deploy > 0]
        rows.append({"model": str(z["model"]), "layer": layer, "num_layers": n_layers,
                     "tokens_per_seq": len(t) / n_seq,
                     "top_channels": int(t.groupby("expert").c1.nunique().max()),
                     "mu_gt_eps_pct": 100 * float((t.mu > t.eps).mean()),
                     "range_ratio": float((ok.range_before / ok.range_deploy).median()),
                     "tokens": len(t)})
    return rows


def main():
    args = parse_args()
    df = pd.DataFrame([r for path in args.input for r in model_rows(path)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("# Massive outlier layers within the first sixth of each network\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
