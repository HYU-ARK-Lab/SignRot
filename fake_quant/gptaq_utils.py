# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import math
import logging
import tqdm
import torch
import utils
import quant_utils
import model_utils
from monkeypatch import (
    IOCatcher,
    BreakpointGroup,
    BreakpointException,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTAQ:
    quantizer: quant_utils.WeightQuantizer

    def __init__(self, layer):
        self.layer = layer
        dev = self.layer.weight.device
        W = layer.weight.data
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=dev)
        self.nsamples = 0
        self.fp_inp = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())
        del self.fp_inp[0]

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        alpha=0.25,
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
        self.dXXT[:, dead] = 0

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
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # scale it by alpha due to collection of dXXT axnd H
        P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in tqdm.trange(
            0, self.columns, blocksize, desc="GPTAQ fasterquant blocks", leave=False
        ):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in tqdm.trange(count, desc="GPTAQ fasterquant cols", leave=False):
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

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                ) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

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
        self.dXXT = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptaq_fwrd(model, dataloader, device, args):
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    logging.info("-----GPTAQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    layers_prefix = "model.layers"

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = first_layer = IOCatcher(layers[0].to(device))

    inps = []
    input_kwargs = {}
    with (
        # accelerator.split_between_processes(dataloader, apply_padding=True) as split_dataloader,
        first_layer.catch_ctx(input=True, once=False, bp=BreakpointGroup()),
    ):
        for batch in dataloader:
            try:
                model(batch[0].to(device))
            except BreakpointException:
                inp_args, inp_kwargs = first_layer.catched_input
                inps.append(inp_args[0].data)
                input_kwargs["attention_mask"] = inp_kwargs["attention_mask"]
                if hasattr(inp_kwargs, "position_embeddings"):
                    input_kwargs["position_embeddings"] = inp_kwargs["position_embeddings"]
                else:
                    input_kwargs["position_ids"] = inp_kwargs["position_ids"]
    inps = torch.cat(inps, dim=0)
    layers[0] = first_layer.module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    del first_layer
    utils.cleanup_memory(verbos=False)

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    tqdm_desc = f"(GPTQ Quant.)"
    for i in (pbar := tqdm.trange(len(layers), desc=f"{tqdm_desc} Layers")):
        pbar.set_description(f"{tqdm_desc} Layer {i}")
        layer = layers[i].to(device)
        # TODO: use IOCacher
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        # bits_config = quant_utils.disable_act_quant(layer)
        fp_inputs_cache.add_hook(full)
        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0), **input_kwargs)[0]
        fp_inputs_cache.clear_hook()
        # quant_utils.enable_act_quant(layer, bits_config)

        for names in sequential:
            gptq: dict[str, GPTAQ] = {}
            for name in names:
                module = layer.get_submodule(name)
                gptq[name] = GPTAQ(module)
                gptq[name].quantizer = create_quantizer(name, module.weight.shape, args)
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            first_module_name = names[0]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                    raise BreakpointException()

                return tmp

            handle = gptq[first_module_name].layer.register_forward_hook(
                add_batch(first_module_name)
            )
            for j in tqdm.trange(
                args.nsamples,
                leave=False,
                desc=f"{tqdm_desc} Layer {i} - catch {names} inputs",
            ):
                try:
                    layer(inps[j].unsqueeze(0), **input_kwargs)
                except BreakpointException:
                    pass
            handle.remove()

            # copy H and dXXT
            for name in names:
                gptq[name].H = gptq[first_module_name].H
                gptq[name].dXXT = gptq[first_module_name].dXXT

            for name in names:
                pbar.set_description(f"{tqdm_desc} Layer {i} - {name}")
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers[f"{layers_prefix}.{i}.{name}"] = gptq[name].quantizer.cpu()
                gptq[name].free()

        if i < len(layers) - 1:
            for j in range(args.nsamples):
                inps[j] = layer(inps[j].unsqueeze(0), **input_kwargs)[0]

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        utils.cleanup_memory(verbos=False)

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTAQ Quantization Done-----\n")
    return quantizers


def create_quantizer(layer_name: str, shape, args):
    w_bits = args.w_bits
    if "lm_head" in layer_name:
        w_bits = 16
    if args.int8_down_proj and "down_proj" in layer_name:
        w_bits = 8
    quantizer = quant_utils.WeightQuantizer(shape)
    quantizer.configure(w_bits, perchannel=True, sym=not (args.w_asym), mse=args.w_clip)
    return quantizer
