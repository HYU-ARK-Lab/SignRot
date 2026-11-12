"""Table activation_distribution_statistics: a massive outlier token vs a non-outlier token."""

import argparse
import os
import sys

import pandas as pd
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "fake_quant"))

import data_utils  # noqa: E402
import hadamard_utils  # noqa: E402
import model_utils  # noqa: E402
import utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--layer", type=int, default=2, help="decoder layer, numbered from 1")
    p.add_argument("--out", required=True, help="output csv (a .md sibling is written too)")
    return p.parse_args()


def stats(v):
    v = v.float()
    return {"mean": v.mean().item(), "var": v.var(unbiased=False).item(), "min": v.min().item(), "max": v.max().item()}


def main():
    args = parse_args()
    transformers.set_seed(0)
    model = model_utils.get_model(args.model, None)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    seq = data_utils.get_loaders("wikitext2", tokenizer, nsamples=1, seed=0, seqlen=2048, eval_mode=False)[0][0]
    model.to(utils.DEV)

    captured = {}
    down = model_utils.get_layers(model)[args.layer - 1].mlp.down_proj
    handle = down.register_forward_pre_hook(lambda _m, inp: captured.setdefault("x", inp[0][0].detach()))
    with torch.inference_mode():
        model(seq.to(utils.DEV))
    handle.remove()

    x = captured["x"]  # (tokens, C_in), model dtype
    had_K, K = hadamard_utils.get_hadK(x.shape[-1])
    y = hadamard_utils.matmul_hadU_cuda(x, had_K, K)
    t_ma = int(y.abs().amax(-1).argmax())
    sign = (y[t_ma] >= 0).to(y.dtype) * 2 - 1
    t_normal = int(x.float().norm(dim=-1).argsort()[x.shape[0] // 2])

    rows = []
    for token, t in (("massive outlier", t_ma), ("non-outlier", t_normal)):
        for name, v in (("Original", x[t]), ("After RHT", y[t]), ("With SignRot", y[t] * sign)):
            rows.append({"token": token, "position": t, "max_abs_x": x[t].abs().max().item(), "row": name, **stats(v)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(f"# {args.model}, layer {args.layer} down_proj input\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
