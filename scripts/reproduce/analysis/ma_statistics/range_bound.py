"""Record mu, eps and the per-token range before/after SignRot of gated massive outlier tokens."""

import argparse
import os
import sys

import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "fake_quant"))

import data_utils  # noqa: E402
import distribute_utils  # noqa: E402
import hadamard_utils  # noqa: E402
import model_utils  # noqa: E402
import utils  # noqa: E402

TRUE_MA_PEAK = 100.0
MASS_FRAC = 0.5
COLS = ("layer", "expert", "sample", "depth", "c1", "mu", "eps", "range_before", "range_deploy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--capture", required=True, help=".pt from capture_topk.py (same model and data)")
    p.add_argument("--out", required=True, help="output .npz of per-token records")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distribute", action="store_true", help="spread the model over all visible GPUs (70B)")
    return p.parse_args()


def site_peaks(capture):
    """{(layer, expert or -1): max_t M_1} of every capture site."""
    data = torch.load(capture, map_location="cpu", weights_only=False, mmap=True)
    peaks = {}
    for e in data["layers"].values():
        if e.get("num_tokens", 0) > 0:
            exp = -1 if e["expert_idx"] is None else int(e["expert_idx"])
            peaks[(int(e["layer_idx"]), exp)] = float(e["topk_values"][:, 0].double().max())
    return peaks


def token_stats(x32):
    """Recursive depth (leading run of rho_j = M_j^2 / sum_{i>j} x_i^2 > 1) and c_1 of each row."""
    order = x32.abs().argsort(dim=-1, descending=True)
    xs2 = torch.gather(x32 * x32, -1, order)
    cum = xs2.cumsum(-1)
    rest = (cum[:, -1:] - cum).clamp_min(1e-30)
    depth = ((xs2 / rest) > 1).int().cumprod(-1).sum(-1)
    return depth, order[:, 0]


def main():
    args = parse_args()
    peaks = site_peaks(args.capture)
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, None)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    loader = data_utils.get_loaders(args.dataset, tokenizer, nsamples=args.nsamples, seed=args.seed,
                                    seqlen=args.seqlen, eval_mode=False)

    bottleneck = model_utils.get_mlp_bottleneck_size(model)
    had_K, K = hadamard_utils.get_hadK(bottleneck)
    targets = []
    for i, layer in enumerate(model_utils.get_layers(model)):
        if model_utils.is_moe_layer(layer):
            targets += [(i, e, exp.down_proj) for e, exp in enumerate(layer.mlp.experts)]
        else:
            targets.append((i, -1, layer.mlp.down_proj))
    # only sites that hold massive outlier tokens produce records
    targets = [t for t in targets if peaks.get((t[0], t[1]), 0.0) >= TRUE_MA_PEAK]
    print(f"{args.model}: {len(targets)} down_proj sites with peak >= {TRUE_MA_PEAK}, C_in={bottleneck}, K={K}")

    signs, hcol, records = {}, {}, []
    state = {"calibrating": True, "sample": 0}

    def h_column(c, device):
        if c not in hcol:
            e = torch.zeros(1, bottleneck, device=device, dtype=torch.float32)
            e[0, c] = 1.0
            h = hadamard_utils.matmul_hadU_cuda(e, had_K, K)[0]
            mag = h.abs()
            assert float(mag.max() - mag.min()) < 1e-6 * float(mag.max()), f"channel {c}: non-uniform column"
            hcol[c] = h
        return hcol[c].to(device)

    def make_hook(li, ei):
        key = (li, ei)

        def hook(_mod, inputs):
            x = inputs[0]
            xb = x[0] if x.dim() == 3 else x  # (tokens, C_in); batch size 1
            if xb.numel() == 0:
                return
            x32 = xb.float()
            y = hadamard_utils.matmul_hadU_cuda(x32, had_K, K)
            if key not in signs:  # deployed calibration: first batch reaching the site
                j = int(y.abs().max(dim=-1).values.argmax())
                signs[key] = (y[j] >= 0).float() * 2 - 1
            if state["calibrating"]:
                return
            peak = peaks[key]
            keep = torch.nonzero(x32.abs().amax(-1) >= MASS_FRAC * peak, as_tuple=True)[0]
            if keep.numel() == 0:
                return
            xk, yk = x32[keep], y[keep]
            depth, c1 = token_stats(xk)
            hc = torch.stack([h_column(int(c), xk.device) for c in c1.tolist()])
            outlier = xk.gather(1, c1[:, None]) * hc
            mu = outlier.abs().amax(-1)
            eps = (yk - outlier).abs().amax(-1)
            y_dep = yk * signs[key][None, :]
            n = keep.numel()
            rec = torch.stack([
                torch.full((n,), li, device=xk.device, dtype=torch.float32),
                torch.full((n,), ei, device=xk.device, dtype=torch.float32),
                torch.full((n,), state["sample"], device=xk.device, dtype=torch.float32),
                depth.float(), c1.float(), mu, eps,
                yk.amax(-1) - yk.amin(-1), y_dep.amax(-1) - y_dep.amin(-1),
            ], dim=1)
            records.append(rec.cpu().numpy())

        return hook

    handles = [mod.register_forward_pre_hook(make_hook(li, ei)) for li, ei, mod in targets]
    if args.distribute:
        distribute_utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    dev = next(model.parameters()).device

    with torch.inference_mode():
        for s in range(args.nsamples):
            state["calibrating"] = s == 0
            state["sample"] = s
            model(loader[s][0].to(dev))
            if (s + 1) % 16 == 0:
                print(f"  {s + 1}/{args.nsamples} samples", flush=True)
    for h in handles:
        h.remove()

    rec = np.concatenate(records, axis=0) if records else np.zeros((0, len(COLS)), np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, rec=rec, cols=np.array(COLS), model=args.model,
                        num_layers=model.config.num_hidden_layers, nsamples=args.nsamples, seqlen=args.seqlen)
    print(f"{len(rec)} gated token records -> {args.out}")


if __name__ == "__main__":
    main()
