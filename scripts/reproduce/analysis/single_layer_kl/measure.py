"""Single-layer quantization KL: final KL when only one decoder layer is quantized (others rotated FP16)."""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../fake_quant"))

import accelerate  # noqa: E402
import transformers  # noqa: E402

import data_utils  # noqa: E402
import main as fq_main  # noqa: E402
import model_utils  # noqa: E402
import rotation_utils  # noqa: E402
import utils  # noqa: E402


def parse_layers(spec, L):
    """'all' or comma-separated 'a-b' (inclusive), 'a:b:c' (range), 'a'."""
    if spec == "all":
        return list(range(L))
    out = set()
    for item in spec.split(","):
        if "-" in item:
            a, b = map(int, item.split("-"))
            out.update(range(a, b + 1))
        elif ":" in item:
            out.update(range(*map(int, item.split(":"))))
        else:
            out.add(int(item))
    return sorted(l for l in out if 0 <= l < L)


def saved_rotation(qmodel_dir):
    """Rotation Q saved (fp16) with the quantized model, rebuilt exactly."""
    for fn in sorted(os.listdir(qmodel_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(qmodel_dir, fn), "pt") as f:
                if "Q" in f.keys():
                    q16 = f.get_tensor("Q")
                    break
    mag = q16.abs()
    assert torch.all(mag == mag[0, 0]), "Q is not a +-c randomized Hadamard"
    return torch.sign(q16.double()) / torch.tensor(q16.shape[0]).sqrt()


def layer_kwargs(model, h):
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    return {"attention_mask": None, "position_ids": pos,
            "position_embeddings": model.model.rotary_emb(h, pos), "use_cache": False}


def run_layer(layer, h, kwargs):
    """Run h [N, S, D] through one decoder layer, one sequence at a time."""
    outs = []
    for s in range(h.shape[0]):
        out = layer(h[s:s + 1].to(utils.DEV), **kwargs)
        outs.append(out[0] if isinstance(out, tuple) else out)
    return torch.cat(outs)


def log_probs(model, h):
    return F.log_softmax(model.lm_head(model.model.norm(h)).float(), dim=-1)


def kl_of(lp, ref_lp):
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1).reshape(-1).cpu()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nseq", type=int, default=8, help="WikiText-2 test windows")
    p.add_argument("--layers", default="all", help="layers quantized one at a time, e.g. 0-7,8:80:4,79")
    cli, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    accelerator = accelerate.Accelerator()
    args = utils.parser_gen()
    assert args.load_qmodel and args.rotate, "needs --rotate and a saved quantized model (--load_qmodel)"
    transformers.set_seed(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)

    Q = saved_rotation(args.load_qmodel)
    rotation_utils.get_orthogonal_matrix = lambda size, mode, device=utils.DEV, seed=0: Q.to(device)

    ref = model_utils.get_model(args.model, args.hf_token)
    ref.eval()
    args_fp = copy.copy(args)
    args_fp.load_qmodel, args_fp.adj_layers = False, []
    fp = model_utils.get_model(args.model, args.hf_token)
    fp.eval()
    fq_main.wrap_model(fp, args_fp, accelerator)
    fp.half()
    q = model_utils.get_model(args.model, args.hf_token)
    q.eval()
    fq_main.wrap_model(q, args, accelerator)
    fq_main.load_qmodel(q, {"": "cpu"}, args)
    fq_main.configure_activation_quantizers(q, args)

    ids = data_utils.get_loaders("wikitext2", tokenizer, seqlen=ref.seqlen, eval_mode=True)[: cli.nseq].to(utils.DEV)
    N, S = ids.shape
    T = N * S
    ref_layers, fp_layers, q_layers = (model_utils.get_layers(m) for m in (ref, fp, q))
    L = len(ref_layers)
    layers = parse_layers(cli.layers, L)
    print(f"[layers] quantizing {len(layers)}/{L} layers one at a time: {layers}", flush=True)

    for m in (ref, fp):
        m.model.embed_tokens.to(utils.DEV)
        m.model.rotary_emb.to(utils.DEV)
    h_ref, h_fp = ref.model.embed_tokens(ids), fp.model.embed_tokens(ids)
    kw_ref, kw_rot = layer_kwargs(ref, h_ref), layer_kwargs(fp, h_fp)
    for m in (ref, fp):
        m.model.embed_tokens.cpu()
    h_full = h_fp.clone()
    # fp.half() rounds the RoPE inv_freq to fp16 (kw_rot); the all-layers run keeps q's fp32 inv_freq
    q.model.rotary_emb.to(utils.DEV)
    kw_full = layer_kwargs(q, h_full)
    is_ma = torch.zeros(T, dtype=torch.bool, device=utils.DEV)
    branch = {}  # l -> hidden states with only layer l quantized

    for j in range(L):
        a = h_ref.reshape(T, -1).abs().float()
        tokmax = a.amax(-1)
        is_ma |= (tokmax > 100) & (tokmax > 1000 * a.median(-1).values)
        del a

        layer = ref_layers[j].to(utils.DEV)
        h_ref = run_layer(layer, h_ref, kw_ref)
        ref_layers[j] = layer.cpu()

        layer = q_layers[j].to(utils.DEV)
        if j in layers:
            branch[j] = run_layer(layer, h_fp, kw_rot).cpu()
        h_full = run_layer(layer, h_full, kw_full)
        q_layers[j] = layer.cpu()

        layer = fp_layers[j].to(utils.DEV)
        for l in branch:
            if l < j:
                branch[l] = run_layer(layer, branch[l], kw_rot).cpu()
        h_fp = run_layer(layer, h_fp, kw_rot)
        fp_layers[j] = layer.cpu()
        print(f"[layer {j:3d}] done", flush=True)
        utils.cleanup_memory(verbos=False)

    kl = torch.full((L, T), float("nan"))
    floor_kl, full_kl = torch.zeros(T), torch.zeros(T)
    for m in (ref, fp):
        m.model.norm.to(utils.DEV)
        m.lm_head.to(utils.DEV)
    for k in range(N):
        sl = slice(k * S, (k + 1) * S)
        r = log_probs(ref, h_ref[k:k + 1])
        floor_kl[sl] = kl_of(log_probs(fp, h_fp[k:k + 1]), r)
        full_kl[sl] = kl_of(log_probs(fp, h_full[k:k + 1]), r)
        for l, hb in branch.items():
            kl[l, sl] = kl_of(log_probs(fp, hb[k:k + 1].to(utils.DEV)), r)

    os.makedirs(args.save_path, exist_ok=True)
    out = os.path.join(args.save_path, "single_layer_kl.npz")
    np.savez_compressed(out, kl=kl.numpy(), floor_kl=floor_kl.numpy(), full_kl=full_kl.numpy(),
                        is_ma=is_ma.cpu().numpy())
    per_layer = kl.nanmean(-1)
    worst = per_layer.nan_to_num(-1).argsort(descending=True)[:5].tolist()
    print(f"[final] MA tokens {int(is_ma.sum())}/{T}; KL rotation only {floor_kl.mean():.4g}, all layers "
          f"quantized {full_kl.mean():.4g}; largest single-layer KL at layers {worst} "
          f"{[round(float(per_layer[l]), 5) for l in worst]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
