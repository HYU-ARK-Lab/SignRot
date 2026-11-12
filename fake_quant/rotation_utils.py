# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import math
import typing
from collections import OrderedDict

from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from fast_hadamard_transform import hadamard_transform
import torch
import transformers
import tqdm

import utils
import model_utils
import quant_utils
from distribute_utils import sync_module, split_with_rank
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2


def add_bias_to_linears(norm, linears):
    for linear in linears:
        if hasattr(norm, "bias") and linear.bias is None:
            linear.bias = torch.nn.Parameter(
                torch.zeros(linear.out_features, dtype=torch.float64)
            )


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            add_bias_to_linears(layernorm, [linear])
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def replace_layer_norms(model, model_type):
    hidden_size = model.config.hidden_size
    for layer in model_utils.get_transformer_layers(model, model_type):
        layer.input_layernorm = model_utils.RMSN(hidden_size)
        layer.post_attention_layernorm = model_utils.RMSN(hidden_size)
    model.model.norm = model_utils.RMSN(hidden_size)


def fuse_embeddings(model, model_type):
    # Embedding fusion
    for W in model_utils.get_embeddings(model, model_type):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)


def fuse_layer_norms(model, device=utils.DEV):
    model_type = model_utils.get_model_type(model)

    # Embedding fusion
    fuse_embeddings(model, model_type)

    layers = model_utils.get_transformer_layers(model, model_type)
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in tqdm.tqdm(layers, unit="layer", desc="Fusing LayerNorms"):
        layer.to(device)
        # fuse the input layernorms into the linear layers
        attn_norm = model_utils.get_attn_layernorm(layer, model_type)
        qkv_linears = model_utils.get_qkv_linears(layer, model_type)
        attn_out_linear = model_utils.get_attn_output_linear(layer, model_type)
        mlp_norm = model_utils.get_mlp_layernorm(layer, model_type)
        mlp_linears = model_utils.get_mlp_input_linears(layer, model_type)
        mlp_out_linear = model_utils.get_mlp_output_linear(layer, model_type)

        fuse_ln_linear(attn_norm, qkv_linears)
        fuse_ln_linear(mlp_norm, mlp_linears)
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(attn_out_linear)
            bake_mean_into_linear(mlp_out_linear)
        layer.cpu()

    pre_head_layernorm = model_utils.get_pre_head_layernorm(model, model_type)
    lm_head = model_utils.get_lm_head(model, model_type)
    fuse_ln_linear(pre_head_layernorm, [lm_head])

    replace_layer_norms(model, model_type)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=utils.DEV, seed=0):
    if seed >= 0:
        torch.random.manual_seed(seed)  # for reproducibility, you can remove this line to get different random orthogonal matrices each time
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.get_model_type(model)
    for W in model_utils.get_embeddings(model, model_type):
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(W.weight)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(W.weight)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_utils.is_llama_like(model_type):
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(W.weight)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(W.bias)


def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_utils.is_llama_like(model_type):
        if model_utils.is_moe_layer(layer):
            mlp_inputs = [layer.mlp.gate]
            for expert in layer.mlp.experts:
                mlp_inputs += [expert.gate_proj, expert.up_proj]
        else:
            mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f"Unknown model type {model_type}")
    for W in mlp_inputs:
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(W.weight)


def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_utils.is_llama_like(model_type):
        if model_utils.is_moe_layer(layer):
            for expert in layer.mlp.experts:
                W = expert.down_proj
                W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
                W.weight.data = torch.matmul(Q.T, W_).to(W.weight)
                apply_exact_had_to_linear(W, had_dim=-1, output=False)
            return
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f"Unknown model type {model_type}")
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(W.weight)
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(W.bias)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(
        model, model_type=model_utils.get_model_type(model)
    )
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(W.weight)


def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_utils.is_llama_like(model_type):
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


def modify_only_for_load_qmodel(model):
    model_type = model_utils.get_model_type(model)
    model_kwargs = {"model": model, "model_type": model_type}
    layers = model_utils.get_transformer_layers(**model_kwargs)

    pre_head_norm = model_utils.get_pre_head_layernorm(**model_kwargs)
    lm_head = model_utils.get_lm_head(**model_kwargs)
    add_bias_to_linears(pre_head_norm, [lm_head])

    for layer in layers:
        attn_norm = model_utils.get_attn_layernorm(layer, model_type)
        mlp_norm = model_utils.get_mlp_layernorm(layer, model_type)
        qkv_linears = model_utils.get_qkv_linears(layer, model_type)
        mlp_linears = model_utils.get_mlp_input_linears(layer, model_type)
        add_bias_to_linears(attn_norm, qkv_linears)
        add_bias_to_linears(mlp_norm, mlp_linears)

    replace_layer_norms(model, model_type)


@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode, seed=args.rotation_seed)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    # save global Q
    model.register_buffer("Q", Q.cpu())
    model_type = model_utils.get_model_type(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


def fuse_and_rotate_model(model, accelerator: Accelerator, args):
    device = accelerator.device
    local_rank = accelerator.process_index
    model_type = model_utils.get_model_type(model)

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode, device, args.rotation_seed)
    # save global Q
    model.register_buffer("Q", Q.cpu())

    # Embedding fusion
    fuse_embeddings(model, model_type)
    rotate_embeddings(model, Q)

    layers = model_utils.get_transformer_layers(model, model_type)
    tqdm_desc = f"[rank {local_rank}] Fuse and rotate layers"
    with split_with_rank(list(range(len(layers)))) as rank_layer_ids:
        split_layer_ids = rank_layer_ids[local_rank]

        for layer_idx in tqdm.tqdm(range(len(layers)), unit="layer", desc=tqdm_desc):
            layer = layers[layer_idx]
            # fuse the input layernorms into the linear layers
            attn_norm = model_utils.get_attn_layernorm(layer, model_type)
            qkv_linears = model_utils.get_qkv_linears(layer, model_type)
            attn_out_linear = model_utils.get_attn_output_linear(layer, model_type)
            mlp_norm = model_utils.get_mlp_layernorm(layer, model_type)
            mlp_linears = model_utils.get_mlp_input_linears(layer, model_type)
            mlp_out_linear = model_utils.get_mlp_output_linear(layer, model_type)

            if layer_idx in split_layer_ids:
                layer.to(device)
                fuse_ln_linear(attn_norm, qkv_linears)
                fuse_ln_linear(mlp_norm, mlp_linears)
                if model_type == model_utils.OPT_MODEL:
                    bake_mean_into_linear(attn_out_linear)
                    bake_mean_into_linear(mlp_out_linear)
                rotate_attention_inputs(layer, Q, model_type)
                rotate_attention_output(layer, Q, model_type)
                rotate_mlp_input(layer, Q, model_type)
                rotate_mlp_output(layer, Q, model_type)
                rotate_ov_proj(layer, model_type, num_heads, head_dim)
                layers[layer_idx] = layer.cpu()
            else:
                add_bias_to_linears(attn_norm, qkv_linears)
                add_bias_to_linears(mlp_norm, mlp_linears)

            del layer
        accelerator.wait_for_everyone()

        if accelerator.num_processes > 1:
            layer_rank_map = [
                (layer_idx, rank)
                for rank, layer_ids in rank_layer_ids.items()
                for layer_idx in layer_ids
            ]
            for layer_idx, rank in tqdm.tqdm(
                layer_rank_map, desc=f"[rank {local_rank}] Sync layers"
            ):
                sync_module(layers[layer_idx], device, rank, {"disable": True})

    pre_head_layernorm = model_utils.get_pre_head_layernorm(model, model_type)
    lm_head = model_utils.get_lm_head(model, model_type)
    fuse_ln_linear(pre_head_layernorm, [lm_head])
    rotate_head(model, Q)

    replace_layer_norms(model, model_type)
    utils.cleanup_memory()


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]


def register_online_rotation(module, Q: torch.Tensor):
    assert not hasattr(module, "Q")
    module.register_buffer(
        "Q", Q.T.to(module.weight.data)
    )  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_extend_zero = kwargs["k_extend_zero"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
                extend_zero=self.k_extend_zero,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module, function_name, *args, **kwargs
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """
    import monkeypatch
    import functools

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
