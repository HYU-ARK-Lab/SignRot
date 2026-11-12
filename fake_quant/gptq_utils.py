# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import math
import logging
from collections import OrderedDict

from accelerate.utils import broadcast, gather_object
import tqdm
import torch
import torch.nn as nn

import utils
import quant_utils
import model_utils
import distribute_utils
from monkeypatch import (
    BreakpointException,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    quantizer: quant_utils.WeightQuantizer

    def __init__(self, layer):
        self.layer = layer
        dev = self.layer.weight.device
        W = layer.weight.data
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        dev = W.device
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in tqdm.trange(
            0, self.columns, blocksize, desc="GPTQ fasterquant blocks", leave=False
        ):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in tqdm.trange(count, desc="GPTQ fasterquant cols", leave=False):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")
            import pprint

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None


def _quantize_moe_mlp(layer, full, quantizers, layers_prefix, layer_idx,
                       inps, input_kwargs, args, accelerator, pbar, tqdm_desc):
    """Quantize all experts in a MoE layer.

    All experts' gate_proj hooks are registered simultaneously so each expert
    captures only the tokens actually routed to it across nsamples full forwards.
    gate_proj and up_proj share the same input, so H is copied gate→up.
    down_proj is captured in a second pass after gate/up are quantized.
    """
    num_experts = len(layer.mlp.experts)

    # ── Step 1: gate_proj + up_proj ─────────────────────────────────────────
    gptq_gate_up: dict[str, GPTQ] = {}
    handles = []
    for i in range(num_experts):
        for proj in ("gate_proj", "up_proj"):
            name = f"mlp.experts.{i}.{proj}.module"
            gptq_gate_up[name] = GPTQ(full[name])
            gptq_gate_up[name].quantizer = create_quantizer(name, full[name].weight.shape, args)

        gate_name = f"mlp.experts.{i}.gate_proj.module"
        def _make_gate_hook(n, gptq_dict):
            def _hook(_, inp):
                gptq_dict[n].add_batch(inp[0].data, None)
            return _hook
        handles.append(full[gate_name].register_forward_pre_hook(_make_gate_hook(gate_name, gptq_gate_up)))

    for j in range(args.nsamples):
        layer(inps[j].unsqueeze(0), **input_kwargs)
    for h in handles:
        h.remove()

    for i in range(num_experts):
        gate_name = f"mlp.experts.{i}.gate_proj.module"
        up_name = f"mlp.experts.{i}.up_proj.module"
        gptq_gate_up[up_name].H = gptq_gate_up[gate_name].H
        for name in (gate_name, up_name):
            pbar.set_description(f"{tqdm_desc} Layer {layer_idx} - {name}")
            if accelerator.is_main_process:
                gptq_gate_up[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
            key = f"{layers_prefix}.{layer_idx}.{name}"
            quantizers[key] = gptq_gate_up[name].quantizer
            if accelerator.num_processes > 1:
                full[name].weight.data = broadcast(full[name].weight.contiguous().to(full[name].weight.device), 0)
                quantizers[key].scale = broadcast(quantizers[key].scale.contiguous().to(full[name].weight.device), 0)
                quantizers[key].zero = broadcast(quantizers[key].zero.contiguous().to(full[name].weight.device), 0)
            quantizers[key] = quantizers[key].cpu()
            gptq_gate_up[name].free()
    del gptq_gate_up
    utils.cleanup_memory(verbos=False)

    # ── Step 2: down_proj (capture after gate/up quantized) ─────────────────
    gptq_down: dict[str, GPTQ] = {}
    handles = []
    for i in range(num_experts):
        name = f"mlp.experts.{i}.down_proj.module"
        gptq_down[name] = GPTQ(full[name])
        gptq_down[name].quantizer = create_quantizer(name, full[name].weight.shape, args)
        def _make_down_hook(n, gptq_dict):
            def _hook(_, inp):
                gptq_dict[n].add_batch(inp[0].data, None)
            return _hook
        handles.append(full[name].register_forward_pre_hook(_make_down_hook(name, gptq_down)))

    for j in range(args.nsamples):
        layer(inps[j].unsqueeze(0), **input_kwargs)
    for h in handles:
        h.remove()

    for i in range(num_experts):
        name = f"mlp.experts.{i}.down_proj.module"
        pbar.set_description(f"{tqdm_desc} Layer {layer_idx} - {name}")
        if accelerator.is_main_process:
            gptq_down[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
        key = f"{layers_prefix}.{layer_idx}.{name}"
        quantizers[key] = gptq_down[name].quantizer
        if accelerator.num_processes > 1:
            full[name].weight.data = broadcast(full[name].weight.contiguous().to(full[name].weight.device), 0)
            quantizers[key].scale = broadcast(quantizers[key].scale.contiguous().to(full[name].weight.device), 0)
            quantizers[key].zero = broadcast(quantizers[key].zero.contiguous().to(full[name].weight.device), 0)
        quantizers[key] = quantizers[key].cpu()
        gptq_down[name].free()
    del gptq_down
    utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, accelerator, args):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    logging.info("-----GPTQ Quantization-----")
    device = accelerator.device
    local_rank = accelerator.process_index

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    layers_prefix = "model.layers"

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0}
    input_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            input_kwargs["attention_mask"] = kwargs["attention_mask"]
            if "position_embeddings" in kwargs:
                input_kwargs["position_embeddings"] = kwargs["position_embeddings"]
            else:
                input_kwargs["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    quantizers = {}
    attn_sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
    ]
    dense_mlp_sequential = [
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    tqdm_desc = f"(GPTQ Quant.)"
    for i in (pbar := tqdm.trange(len(layers), desc=f"{tqdm_desc} Layers")):
        pbar.set_description(f"{tqdm_desc} Layer {i}")
        layer = layers[i].to(device)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        sequential = attn_sequential + ([] if model_utils.is_moe_layer(layer) else dense_mlp_sequential)
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq: dict[str, GPTQ] = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = create_quantizer(name, subset[name].weight.shape, args)

            first_module_name = names[0]

            def add_batch(name):
                def tmp(_, inp):
                    gptq[name].add_batch(inp[0].data, None)
                    # raise BreakpointException()

                return tmp

            handle = gptq[first_module_name].layer.register_forward_pre_hook(
                add_batch(first_module_name)
            )
            for j in range(args.nsamples):
                try:
                    layer(inps[j].unsqueeze(0), **input_kwargs)
                except BreakpointException:
                    pass
            handle.remove()

            # copy H and dXXT
            for name in names:
                gptq[name].H = gptq[first_module_name].H

            for name in subset:
                pbar.set_description(f"{tqdm_desc} Layer {i} - {name}")
                layer_w_groupsize = args.w_groupsize
                if accelerator.is_main_process:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=layer_w_groupsize,
                        actorder=args.act_order,
                        static_groups=False,
                    )

                key = f"{layers_prefix}.{i}.{name}"
                quantizers[key] = gptq[name].quantizer
                if accelerator.num_processes > 1:
                    main_rank = 0
                    subset[name].weight.data = broadcast(subset[name].weight.contiguous().to(device), main_rank)
                    quantizers[key].scale = broadcast(quantizers[key].scale.contiguous().to(device), main_rank)
                    quantizers[key].zero = broadcast(quantizers[key].zero.contiguous().to(device), main_rank)
                quantizers[key] = quantizers[key].cpu()

                gptq[name].free()
            del gptq
            utils.cleanup_memory(verbos=False)

        if model_utils.is_moe_layer(layer):
            _quantize_moe_mlp(
                layer, full, quantizers, layers_prefix, i,
                inps, input_kwargs, args, accelerator, pbar, tqdm_desc,
            )

        for j in range(args.nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), **input_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        utils.cleanup_memory(verbos=False)
    accelerator.wait_for_everyone()


    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, accelerator, args):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    device = accelerator.device
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    layers_prefix = "model.layers"
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(device)

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            W = subset[name].weight.data
            w_dtype = W.dtype
            quantizer = create_quantizer(name, W.shape, args)
            # If a SignRot sign was already folded into W (adj_cal_dataset), unfold it
            # before quantization and refold after: the w_clip mse search on W*sign is
            # biased toward the massive-token direction (int4 grid [-8,7] asymmetry),
            # while quantizing the unfolded W keeps RTN independent of the fold timing.
            sign = None
            if name.endswith(".module"):
                parent = layer.get_submodule(name.rsplit(".", 1)[0])
                if (
                    isinstance(parent, quant_utils.ActQuantWrapper)
                    and parent.had_adj_sign is not None
                    and parent.had_adj_ready
                ):
                    sign = parent.had_adj_sign.to(device=W.device, dtype=W.dtype)
            if sign is not None:
                W = W * sign
                # refolding flips columns with sign -1, where an int of -8 would land
                # on +8 — unrepresentable on the int4 grid [-8, 7] (kernel pack asserts,
                # packed export clamps). Search and round on the symmetric grid [-7, 7]
                # so the folded weight stays exactly representable (fake == real quant).
                quantizer.sym_symmetric_grid = True
            quantizer.find_params(W)
            Wq = quantizer(W)
            if sign is not None:
                Wq = Wq * sign
            subset[name].weight.data = Wq.to(w_dtype)
            quantizers[f"{layers_prefix}.{i}.{name}"] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers

@torch.no_grad()
def rtn_fwrd_parallel(model, accelerator, args):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    device = accelerator.device
    local_rank = accelerator.process_index

    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    layers_prefix = "model.layers"
    torch.cuda.empty_cache()

    quantizers = {}
    tqdm_desc = f"[rank {local_rank}] (RtN Quant.)"
    with (
        accelerator.split_between_processes(list(range(len(layers)))) as split_layer_ids,
        tqdm.tqdm(split_layer_ids, desc=f"{tqdm_desc} Layers") as pbar,
    ):
        layer_rank_map = OrderedDict(gather_object([(idx, local_rank) for idx in split_layer_ids])) # type: ignore
        
        for layer_idx, layer in enumerate(layers):
            layer_rank = layer_rank_map[layer_idx]
            if layer_rank == local_rank:
                layer.to(device)
            subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

            for name in subset:
                W = subset[name].weight.data
                w_dtype = W.dtype
                quantizer = create_quantizer(name, W.shape, args)
                if layer_rank == local_rank:
                    quantizer.find_params(W)
                    subset[name].weight.data = quantizer(W).to(w_dtype)
                quantizers[f"{layers_prefix}.{layer_idx}.{name}"] = quantizer
                # quantizers[f"{layers_prefix}.{layer_idx}.{name}"] = quantizer.cpu()
            # layer.cpu()
            torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    if accelerator.num_processes > 1:
        for layer_idx, rank in tqdm.tqdm(layer_rank_map.items(), desc=f"[rank {local_rank}] Sync layers"):
            subset = quant_utils.find_qlayers(layers[layer_idx], layers=[torch.nn.Linear])
            for name in subset:
                key = f"{layers_prefix}.{layer_idx}.{name}"
                subset[name].weight.data = broadcast(subset[name].weight.contiguous().to(device), rank)
                quantizers[key].scale = broadcast(quantizers[key].scale.contiguous().to(device), rank)
                quantizers[key].zero = broadcast(quantizers[key].zero.contiguous().to(device), rank)

    utils.cleanup_memory(verbos=True)
    return quantizers


def create_quantizer(layer_name: str, shape, args):
    w_bits = args.w_bits
    if "lm_head" in layer_name:
        w_bits = 16
    if args.int8_down_proj and "down_proj" in layer_name:
        w_bits = 8
    quantizer = quant_utils.WeightQuantizer(shape)
    quantizer.configure(
        w_bits,
        perchannel=True,
        sym=not (args.w_asym),
        mse=args.w_clip,
        extend_zero=args.w_extend_zero,
    )
    return quantizer


@torch.no_grad()
def adjust_hadamard(model, dataloader, accelerator, args):
    """Find massive outlier channel and adjust Hadamard matrix to reduce outlier effect
    """
    logging.info("-----Adjust Hadamard matrix-----")
    device = accelerator.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dataloader = dataloader[:1]

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (1, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0}
    input_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            input_kwargs["attention_mask"] = kwargs["attention_mask"]
            if "position_embeddings" in kwargs:
                input_kwargs["position_embeddings"] = kwargs["position_embeddings"]
            else:
                input_kwargs["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    if args.log_ma_position:
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in [1, 31]:
                continue
            qlayers: dict[str, quant_utils.ActQuantWrapper] = quant_utils.find_qlayers(layer, [quant_utils.ActQuantWrapper])  # type: ignore
            for name in qlayers:
                qlayers[name].label = name
                qlayers[name].log_ma_position = True

    for i in tqdm.trange(len(layers), desc="(Hadamard Adjust) Layers"):
        layer = layers[i].to(device)
        for j in range(len(dataloader)):
            inps[j] = layer(inps[j].unsqueeze(0), **input_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        utils.cleanup_memory(verbos=False)
    del inps

    if args.log_ma_position:
        import pprint
        ma_log_dict = {}
        ma_logs_before_rotation_dict = {}
        distribution_shift_log_dict = {}
        distribution_shift_before_adjust_log_dict = {}
        for layer_idx, layer in enumerate(layers):
            qlayers: dict[str, quant_utils.ActQuantWrapper] = quant_utils.find_qlayers(layer, [quant_utils.ActQuantWrapper])  # type: ignore
            for name in qlayers:
                if qlayers[name].log_ma_position:
                    ma_log_dict.setdefault(name, {})[layer_idx] = qlayers[name].ma_logs
                    ma_logs_before_rotation_dict.setdefault(name, {})[layer_idx] = qlayers[name].ma_logs_before_rotation
                    distribution_shift_before_adjust_log_dict.setdefault(name, {})[layer_idx] = qlayers[name].distribution_shift_before_adjust_logs
                    distribution_shift_log_dict.setdefault(name, {})[layer_idx] = qlayers[name].distribution_shift_logs
                    qlayers[name].log_ma_position = False
                    qlayers[name].ma_logs = []
                    qlayers[name].ma_logs_before_rotation = []
                    qlayers[name].distribution_shift_before_adjust_logs = []
                    qlayers[name].distribution_shift_logs = []
        logging.info(f"MA positions before rotation:\n{pprint.pformat(ma_logs_before_rotation_dict)}")
        logging.info(f"MA positions:\n{pprint.pformat(ma_log_dict)}")
        logging.info(f"Distribution shift before hadamard adjustment:\n{pprint.pformat(distribution_shift_before_adjust_log_dict)}")
        logging.info(f"Distribution shift after hadamard adjustment:\n{pprint.pformat(distribution_shift_log_dict)}")
        del ma_log_dict, ma_logs_before_rotation_dict, distribution_shift_log_dict, distribution_shift_before_adjust_log_dict

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----Adjust Hadamard matrix Done-----\n")
