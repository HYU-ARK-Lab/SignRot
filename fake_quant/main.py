# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import os
import logging
import functools
import warnings

from accelerate import Accelerator, dispatch_model, infer_auto_device_map
from accelerate.hooks import attach_align_device_hook
from accelerate.utils import get_max_memory, get_balanced_memory
import torch
import transformers

import utils
import model_utils
import bench_utils
import data_utils
import kernel_utils
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
from realquant_utils import (
    save_quantized_weights_with_safetensors,
    load_quantized_weights_with_safetensors,
)
from distribute_utils import group_gpu


warnings.filterwarnings("ignore", message=".*`_is_quantized_training_enabled`")
warnings.filterwarnings("ignore", message=".*`resume_download`")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node")


def main():
    accelerator = Accelerator()
    local_rank = accelerator.process_index
    args = utils.parser_gen(local_rank)

    logging.info(f"Process {local_rank} using device: {accelerator.device}")
    logging.info(f"Available GPUs: {torch.cuda.device_count()}")

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=False, use_auth_token=args.hf_token
    )

    # Rotate the weights
    wrap_model(model, args, accelerator)

    quantizers = None
    if args.load_qmodel:  # Load Quantized Rotated Model
        # assert args.rotate, "Model should be rotated to load a quantized model!"
        assert not args.save_qmodel, (
            "Cannot save a quantized model if it is already loaded!"
        )
    else:
        # gptq calibration 이후 Hadamard Adjustment가 진행되면 gptq 오차 보정이 망가질 수 있음
        assert not ("v_proj" in args.adj_layers and args.adj_cal_dataset is None), (
            "Hadamard Adjustment requires both adj_layers and adj_cal_dataset to be specified! if w_quantizer is gptq, perform Hadamard Adjustment after gptq calibration will break the gptq calibration!"
        )
        if args.adj_cal_dataset is not None and len(args.adj_layers) > 0:
            logging.info(
                f"Calibrating Hadamard Adjustment with {args.adj_cal_dataset} dataset..."
            )
            cal_loader = data_utils.get_loaders(
                args.adj_cal_dataset,
                tokenizer,
                nsamples=1,
                seed=args.adj_cal_dataset_seed,
                eval_mode=False,
            )
            gptq_utils.adjust_hadamard(model, cal_loader[:1], accelerator, args)

        if args.w_bits < 16:
            quantizers = quantize_model(model, tokenizer, args, accelerator)

            if args.save_qmodel and accelerator.is_main_process:
                save_qmodel(model, quantizers, args)
            utils.cleanup_memory(verbos=True)
            accelerator.wait_for_everyone()

    if args.runtime_bench:
        # dedicated single-process path: the shared eval path below sizes the
        # device map from the fp16 footprint, which does not fit small GPUs
        assert (
            accelerator.num_processes == 1
        ), "--runtime_bench only supports single-process runs"
        if args.load_qmodel:
            # The eval path below loads into an fp16-sized device map; here the
            # weights are staged on CPU instead and dispatch_for_bench places
            # the model once it has been packed to int4.
            quantizers = load_qmodel(model, {"": "cpu"}, args)
        configure_activation_quantizers(model, args)
        if args.use_kernel and quantizers is not None:
            enable_kernel_modules(model, quantizers, free_fp16=True)
            kernel_utils.patch_kernel_model(model, args)
        utils.cleanup_memory(verbos=True)
        bench_utils.dispatch_for_bench(model)
        results = bench_utils.runtime_bench(model, args)
        logging.info(f"Runtime benchmark results: {results}")
        return

    if not args.ppl_eval and not args.lm_eval:
        return

    process_max_memory = group_gpu(
        model, accelerator.num_processes, args.distribute_mode
    )
    logging.info(f"[rank {local_rank}] Group GPU Map: {process_max_memory}")
    print(f"[rank {local_rank}] Group GPU Map: {process_max_memory}", flush=True)
    device_map = None
    if local_rank in process_max_memory:
        max_memory = process_max_memory[local_rank]
        device_map = infer_auto_device_map(
            model, max_memory, model._no_split_modules, dtype=torch.float16
        )
        if args.load_qmodel:
            quantizers = load_qmodel(model, device_map, args)

    # setting input activation
    configure_activation_quantizers(model, args)

    if args.use_kernel and quantizers is not None:
        enable_kernel_modules(model, quantizers)
        kernel_utils.patch_kernel_model(model, args)

    if local_rank in process_max_memory and device_map is not None:
        dispatch_model(model, device_map)
    group_main_ranks = list(process_max_memory.keys())

    if args.ppl_eval:
        for eval_dataset in args.eval_datasets:
            # Evaluating on dataset
            testloader = data_utils.get_loaders(
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True,  # type: ignore
            )
            logging.info(f"Evaluating {eval_dataset} PPL...")
            # ctx = None
            # if not args.load_qmodel:
            #     ctx = utils.detect_initial_nan(model)
            #     ctx.__enter__()
            dataset_ppl, nlls = eval_utils.ppl_eval(
                model, testloader, args.ppl_eval_bsz, group_main_ranks
            )
            # if ctx is not None:
            #     ctx.__exit__(None, None, None)
            logging.info(f"{eval_dataset} NLL: {nlls}")
            logging.info(f"{eval_dataset} PPL: {dataset_ppl}")
            utils.cleanup_memory(verbos=True)
            accelerator.wait_for_everyone()

        # # save newly calculated had_adj_sign
        # if accelerator.is_main_process:
        #     state_dict = model.state_dict()  # make sure all buffers are synced
        #     had_adj_sign_dict = {k: v for k, v in state_dict.items() if "had_adj_sign" in k}
        #     torch.save(had_adj_sign_dict, f"{args.save_log_dir}/had_adj_sign.pt")

    if not args.lm_eval:
        return

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,  # type: ignore
    )

    # manual setup for distributed eval
    # hflm dose not support pre-distributed model evaluation
    if accelerator.num_processes > 1:
        hflm.accelerator = accelerator  # type: ignore
        hflm._rank = local_rank
        hflm._world_size = len(group_main_ranks)

    raw_results = {}
    results = {}
    for task_name in args.tasks:
        logging.info(f"Evaluating {task_name}...")
        result = simple_evaluate(
            hflm,
            tasks=[task_name],
            batch_size=args.lm_eval_batch_size,
            cache_requests=True,
        )  # type: ignore
        if result is not None:  # same as local_rank == 0
            result = result["results"][task_name]
            raw_results[task_name] = result
            acc = result.get("acc_norm,none", result["acc,none"]) * 100
            results[task_name] = acc
            logging.info(f"{task_name} acc: {acc}%")

    if len(results) > 0:
        logging.info(f"lm_eval raw results: {raw_results}")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals["acc_avg"] = sum(metric_vals.values()) / len(metric_vals.values())
        logging.info(f"lm_eval metric: {metric_vals}")


def get_rotated_cache_path(args):
    tag = f"{args.rotate_mode}_{args.rotation_seed}"
    return os.path.join("experiments", args.model, f"{tag}.pt")


def wrap_model(model, args, accelerator):
    if args.rotate:
        if args.load_qmodel:
            rotation_utils.modify_only_for_load_qmodel(model)
        # elif args.cache_rotated and os.path.exists(get_rotated_cache_path(args)):
        #     cache_path = get_rotated_cache_path(args)
        #     logging.info(f"Loading rotated model from cache: {cache_path}")
        #     rotation_utils.modify_only_for_load_qmodel(model)
        #     model.register_buffer("Q", torch.empty(model.config.hidden_size, model.config.hidden_size))  # type: ignore
        #     model.load_state_dict(torch.load(cache_path, weights_only=True))
        else:
            rotation_utils.fuse_and_rotate_model(model, accelerator, args)
            # if args.cache_rotated and accelerator.is_main_process:
            #     cache_path = get_rotated_cache_path(args)
            #     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            #     torch.save(model.state_dict(), cache_path)
            #     logging.info(f"Saved rotated model cache: {cache_path}")

        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model.model)  # Add Activation Wrapper to the model

        for layer_idx, decoder_layer in enumerate(model_utils.get_layers(model)):
            qlayers: dict[str, quant_utils.ActQuantWrapper] = quant_utils.find_qlayers(
                decoder_layer, layers=[quant_utils.ActQuantWrapper]
            )  # type: ignore
            for name in qlayers:
                if "down_proj" in name:
                    if model_utils.is_moe_layer(decoder_layer):
                        bottleneck = model.config.moe_intermediate_size
                    else:
                        bottleneck = model.config.intermediate_size
                    had_K, K = hadamard_utils.get_hadK(bottleneck)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if "o_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = (
                        model.config.hidden_size // model.config.num_attention_heads
                    )
                    qlayers[name].fp32_had = args.fp32_had
                if args.adj_layer_ids == "all" or str(layer_idx) in args.adj_layer_ids:
                    if any(adj_layer in name for adj_layer in args.adj_layers):
                        layer_dim: int = qlayers[name].weight.shape[1]  # type: ignore
                        qlayers[name].had_adj_sign = torch.ones(
                            layer_dim, device=utils.DEV, dtype=torch.int8
                        )
                        qlayers[name].had_adj_ready = bool(args.load_qmodel)
    else:
        quant_utils.add_actquant(
            model.model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present


def configure_activation_quantizers(model, args):
    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers: dict[str, quant_utils.ActQuantWrapper] = quant_utils.find_qlayers(
            model, layers=[quant_utils.ActQuantWrapper]
        )  # type: ignore
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.get_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            layer_a_extend_zero = args.a_extend_zero

            # with --use_kernel the V-cache is quantized inside the kernel KV cache
            if "v_proj" in name and args.v_bits < 16 and not args.use_kernel:  # Set the v_proj precision
                v_groupsize = args.v_groupsize
                if v_groupsize == "head":
                    v_groupsize = (
                        model.config.hidden_size // model.config.num_attention_heads
                    )
                else:
                    v_groupsize = int(v_groupsize)

                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                    extend_zero=args.v_extend_zero,
                )

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if (
                "mlp.gate" in name and "gate_proj" not in name
            ):  # Skip MoE router quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                extend_zero=layer_a_extend_zero,
            )

    # with --use_kernel the K-cache Hadamard + quantization happen inside the
    # kernel KV cache (and on q in the flashinfer decode stub), so the fake
    # QK rotation wrapper must not be installed
    if args.k_bits < 16 and not args.use_kernel:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)

            k_groupsize = args.k_groupsize
            if k_groupsize == "head":
                k_groupsize = (
                    model.config.hidden_size // model.config.num_attention_heads
                )
            else:
                k_groupsize = int(k_groupsize)

            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
                "k_extend_zero": args.k_extend_zero,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )


def enable_kernel_modules(model, quantizers, free_fp16=False):
    # Replace the fake quantize->linear path of each wrapped linear with the
    # real signrot CUDA kernels, using the calibrated weight quantizer scales.
    qlayers: dict[str, quant_utils.ActQuantWrapper] = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])  # type: ignore
    enabled = 0
    for name, wrapper in qlayers.items():
        if wrapper.quantizer.bits >= 16:
            continue
        # fresh-quantized keys end with ".module", loaded (safetensors) keys do not
        weight_quantizer = quantizers.get(f"{name}.module", quantizers.get(name))
        if weight_quantizer is None:
            logging.warning(f"No weight quantizer for {name}; keeping fake-quant path")
            continue
        wrapper.enable_kernel(weight_quantizer, free_fp16=free_fp16)
        enabled += 1
    logging.info(f"Replaced fake-quant with signrot kernels in {enabled} linear layers")


def quantize_model(model, tokenizer, args, accelerator):
    if args.w_quantizer == "rtn":  # RTN Weight Quantization
        quantizers = gptq_utils.rtn_fwrd(model, accelerator, args)
    else:  # GPTQ Weight Quantization
        logging.info(f"Loading calibration dataset {args.cal_dataset}...")
        trainloader = data_utils.get_loaders(
            args.cal_dataset,
            tokenizer,
            nsamples=args.nsamples,
            seed=args.seed,
            eval_mode=False,
        )
        if args.w_quantizer == "gptq":  # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, accelerator, args)
        # elif args.w_quantizer == "gptaq":  # GPTAQ Weight Quantization
        #     quantizers = gptaq_utils.gptaq_fwrd(model, trainloader, accelerator, args)
        else:
            raise NotImplementedError(f"Unknown weight quantizer {args.w_quantizer}")

    return quantizers


def save_qmodel(model, quantizers, args):
    os.makedirs(args.save_qmodel, exist_ok=True)
    if args.packed_safetensors:
        state_dict = model.state_dict()
        config = model.config.to_diff_dict()
        # map act_quantizer wrapped .module.weight to .weight

        for name in list(state_dict.keys()):
            if name.endswith(".weight") or name.endswith(".bias"):
                atoms = name.split(".")
                # check parent module is quant_utils.ActQuantWrapper
                module = model.get_submodule(".".join(atoms[:-2]))
                if isinstance(module, quant_utils.ActQuantWrapper):
                    new_name = name.replace(".module.weight", ".weight").replace(
                        ".module.bias", ".bias"
                    )
                    state_dict[new_name] = state_dict[name]
                    del state_dict[name]

        save_quantized_weights_with_safetensors(state_dict, quantizers, config, args)
        del state_dict, config
    else:
        qmodel_path = os.path.join(args.save_qmodel, "qmodel.pt")
        torch.save(model.state_dict(), qmodel_path)

        # save weight quantizers for debugging
        quantizers_path = os.path.join(args.save_qmodel, "quantizers.pt")
        torch.save(quantizers, quantizers_path)


def load_qmodel(model, device_map, args):
    logging.info(f"Load quantized model from {args.load_qmodel}")
    quantizers = None
    if args.packed_safetensors:
        state_dict, quantizers = load_quantized_weights_with_safetensors(
            model, args.load_qmodel, dequantize=True, device_map=device_map
        )

        # remap act_quantizer wrapped .weight to .module.weight
        for name in list(state_dict.keys()):
            if name.endswith(".weight") or name.endswith(".bias"):
                module = model.get_submodule(name.rsplit(".", 1)[0])
                if isinstance(module, quant_utils.ActQuantWrapper):
                    new_name = name.replace(".weight", ".module.weight").replace(
                        ".bias", ".module.bias"
                    )
                    state_dict[new_name] = state_dict[name]
                    del state_dict[name]

        model.load_state_dict(state_dict, strict=False, assign=True)
        model.hf_device_map = dict(device_map)  # type: ignore
    else:
        qmodel_path = os.path.join(args.load_qmodel, "qmodel.pt")
        state_dict = torch.load(qmodel_path, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    utils.cleanup_memory(verbos=True)
    return quantizers


if __name__ == "__main__":
    main()
