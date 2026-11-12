"""Capture top-10 |x|, median |x| and sum of squares of every down_proj input token (FP16 model)."""

import argparse
import logging
import os
import sys

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "fake_quant"))

import data_utils  # noqa: E402
import model_utils  # noqa: E402
import utils  # noqa: E402
from distribute_utils import distribute_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="output .pt")
    p.add_argument("--dataset", default="wikitext2", choices=utils.supported_datasets)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distribute", action="store_true", help="spread the model over all visible GPUs (70B)")
    return p.parse_args()


def capture_sites(model):
    """(layer_idx, expert_idx or None, down_proj) for every capture site."""
    for layer_idx, layer in enumerate(model_utils.get_layers(model)):
        if model_utils.is_moe_layer(layer):
            for expert_idx, expert in enumerate(layer.mlp.experts):
                yield layer_idx, expert_idx, expert.down_proj
        else:
            yield layer_idx, None, layer.mlp.down_proj


def make_hook(store, topk):
    def hook(_module, inputs):
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).abs().float()
        if x.shape[0] == 0:  # MoE expert that received no token
            return
        store["topk_values"].append(torch.topk(x, min(topk, x.shape[-1]), dim=-1).values.half().cpu())
        store["median_abs"].append(x.median(dim=-1).values.half().cpu())
        store["sq_sum"].append(x.square().sum(dim=-1).cpu())

    return hook


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    transformers.set_seed(args.seed)

    model = model_utils.get_model(args.model, None)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    loader = data_utils.get_loaders(args.dataset, tokenizer, nsamples=args.nsamples, seed=args.seed,
                                    seqlen=args.seqlen, eval_mode=False)
    if args.distribute:
        distribute_model(model)
    else:
        model.to(utils.DEV)
    device = next(model.parameters()).device

    stores, handles = {}, []
    for layer_idx, expert_idx, module in capture_sites(model):
        store = {"layer_idx": layer_idx, "expert_idx": expert_idx, "dim": module.in_features,
                 "topk_values": [], "median_abs": [], "sq_sum": []}
        key = f"L{layer_idx:03d}" if expert_idx is None else f"L{layer_idx:03d}.E{expert_idx:03d}"
        stores[key] = store
        handles.append(module.register_forward_pre_hook(make_hook(store, args.topk)))

    with torch.inference_mode():
        for s, (inp, _) in enumerate(loader):
            model(inp.to(device))
            logging.info(f"sample {s + 1}/{args.nsamples}")
    for h in handles:
        h.remove()

    layers = {}
    for key, store in stores.items():
        site = {k: store[k] for k in ("layer_idx", "expert_idx", "dim")}
        site["num_tokens"] = sum(len(v) for v in store["topk_values"])
        if site["num_tokens"]:
            for k in ("topk_values", "median_abs", "sq_sum"):
                site[k] = torch.cat(store[k])
        layers[key] = site
    meta = {"model": args.model, "dataset": args.dataset, "nsamples": args.nsamples, "seqlen": args.seqlen,
            "topk": args.topk, "seed": args.seed, "num_layers": model.config.num_hidden_layers}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"meta": meta, "layers": layers}, args.output)
    logging.info(f"saved {len(layers)} capture sites to {args.output}")


if __name__ == "__main__":
    main()
