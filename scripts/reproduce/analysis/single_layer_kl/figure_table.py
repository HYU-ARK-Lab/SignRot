"""Appendix figure (single-layer KL per layer) and table (KL at the massive outlier layers) from measure.py outputs."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e0", "#ecebe6"
NAMES = {"meta-llama/Llama-2-7b-hf": "LLaMA2-7B", "meta-llama/Llama-2-13b-hf": "LLaMA2-13B",
         "meta-llama/Llama-2-70b-hf": "LLaMA2-70B", "meta-llama/Meta-Llama-3-8B": "LLaMA3-8B",
         "meta-llama/Meta-Llama-3-70B": "LLaMA3-70B"}
METHODS = ("quarot", "signrot")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ma_statistics", default=os.path.join(REPO, "fake_quant/experiments/reproduce/ma_statistics/ma_statistics.csv"),
                   help="output of scripts/reproduce/ma_statistics.sh (massive outlier layers: ours_tokens_per_seq > 0)")
    p.add_argument("--exp_root", default=os.path.join(REPO, "fake_quant/experiments"))
    p.add_argument("--out_dir", default=os.path.join(REPO, "fake_quant/experiments/reproduce/single_layer_kl"))
    p.add_argument("--models", nargs="+", default=list(NAMES))
    p.add_argument("--figure_models", nargs="+",
                   default=["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"])
    return p.parse_args()


def load(exp_root, model):
    """Per-layer KL (all / non-MA tokens) per method, and the rotation-only / all-layers KL."""
    rows, extra = {}, {}
    for method in METHODS:
        d = np.load(os.path.join(exp_root, model, "single_layer_kl", method, "single_layer_kl.npz"))
        depth = len(d["kl"])
        measured = np.flatnonzero(~np.isnan(d["kl"]).all(1))
        kl_l = d["kl"][measured]
        rows[(method, "all")] = pd.Series(kl_l.mean(1), index=measured)
        rows[(method, "non_ma")] = pd.Series(kl_l[:, ~d["is_ma"]].mean(1), index=measured)
        extra[method] = (float(d["floor_kl"].mean()), float(d["full_kl"].mean()))
    return pd.DataFrame(rows).dropna(), depth, extra


def main():
    args = parse_args()
    ma = pd.read_csv(args.ma_statistics)
    ma = ma[ma.ours_tokens_per_seq > 0]

    per_layer, table, sums, panels = [], [], [], {}
    for model in args.models:
        kl, depth, extra = load(args.exp_root, model)
        q, s = kl[("quarot", "all")], kl[("signrot", "all")]
        qn, sn = kl[("quarot", "non_ma")], kl[("signrot", "non_ma")]
        sparse = len(kl) < depth
        all_ma = sorted(int(l) - 1 for l in ma[ma.model == model].layer)  # 1-based in the csv
        ma_layers = [l for l in all_ma if l in kl.index]
        panels[model] = (q, s, ma_layers, depth, sparse)
        for l in kl.index:
            per_layer.append({"model": model, "layer": l, "ma_layer": l in ma_layers,
                              "quarot_all": q[l], "signrot_all": s[l], "quarot_non_ma": qn[l], "signrot_non_ma": sn[l]})

        rank = q.rank(ascending=False).astype(int)
        med = q[~q.index.isin(ma_layers)].median()
        for l in ma_layers[:1] if sparse else ma_layers:
            table.append({"Model": NAMES.get(model, model), "Layer": l + 1, "Rank": int(rank[l]),
                          "of": len(q), "QuaRot": f"{q[l]:.4f}", "QuaRot+SignRot": f"{s[l]:.4f}",
                          "Δ all (%)": f"{100 * (s[l] / q[l] - 1):+.1f}",
                          "Δ others (%)": f"{100 * (sn[l] / qn[l] - 1):+.1f}",
                          "x median of the other layers": f"{q[l] / med:.2f}"})
        floor, full = extra["quarot"]
        sums.append({"Model": NAMES.get(model, model), "layers measured": f"{len(q)}/{depth}",
                     "sum QuaRot": f"{q.sum():.4f}", "sum QuaRot+SignRot": f"{s.sum():.4f}",
                     "Δ sum (%)": f"{100 * (s.sum() / q.sum() - 1):+.1f}",
                     "KL all layers / sum of single layers (QuaRot)":
                         "" if sparse else f"{full / (q.sum() - depth * floor):.2f}"})

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(per_layer).to_csv(os.path.join(args.out_dir, "per_layer.csv"), index=False)
    with open(os.path.join(args.out_dir, "table.md"), "w") as f:
        f.write("# Single-layer quantization KL at the massive outlier layers (GPTQ W4A4)\n\n"
                + pd.DataFrame(table).to_markdown(index=False, disable_numparse=True) + "\n\n"
                + "## Sum over the measured layers\n\n" + pd.DataFrame(sums).to_markdown(index=False, disable_numparse=True) + "\n")

    models = [m for m in args.figure_models if m in panels]
    fig, axes = plt.subplots(len(models), 1, figsize=(7, 2.2 * len(models)), constrained_layout=True, squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        q, s, ma_layers, depth, sparse = panels[model]
        for l in ma_layers:
            ax.axvspan(l + 0.5, l + 1.5, color=BAND, lw=0, zorder=0)
        mk = dict(marker="o", ms=3) if sparse else {}
        ax.plot(q.index + 1, q.values, color=BLUE, lw=2, label="QuaRot", zorder=3, **mk)
        ax.plot(s.index + 1, s.values, color=ORANGE, lw=2, label="QuaRot+SignRot", zorder=3, **mk)
        title = NAMES.get(model, model) + (f"  ({len(q)} of {depth} layers measured)" if sparse else "")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0.5, depth + 0.5)
        ax.set_ylim(0, max(q.max(), s.max()) * 1.1)
        ax.set_ylabel("KL", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0, 0].legend(loc="upper center", frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("quantized decoder layer (all others: rotated FP16)", color=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"single_layer_kl.{ext}"), **({"dpi": 200} if ext == "png" else {}))
    print(open(os.path.join(args.out_dir, "table.md")).read())
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
