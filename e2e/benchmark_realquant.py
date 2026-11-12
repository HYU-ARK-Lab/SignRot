import argparse
import logging
import warnings
import os
import sys
from datetime import datetime
import pprint

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "fake_quant"))

from accelerate import Accelerator, dispatch_model
from accelerate.utils import infer_auto_device_map
import torch
import torch.distributed as dist
import transformers

from fake_quant import utils, data_utils, eval_utils
from fake_quant.distribute_utils import group_gpu
from e2e.quantized_llama import modeling_llama

warnings.filterwarnings("ignore", message=".*`_is_quantized_training_enabled`")
warnings.filterwarnings("ignore", message=".*`resume_download`")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node")

# torch.serialization.add_safe_globals([quant_utils.WeightQuantizer])

QUANT_CONFIG_KEYS = (
    "adj_layers",
    "a_asym",
    "a_clip_ratio",
    "k_clip_ratio",
    "v_clip_ratio",
)


def main():
    accelerator = Accelerator()
    local_rank = accelerator.process_index
    args = parser_gen(local_rank)
    assert args.ppl_eval or args.lm_eval, "No evaluation specified!"

    quantization_config = {
        key: value
        for key in QUANT_CONFIG_KEYS
        if (value := getattr(args, key, None)) is not None
    }

    # load and distribute model
    model = modeling_llama.SignrotLlamaForCausalLM.from_pretrained(
        args.qmodel_path, quantization_config
    )
    logging.info(f"---> Loading quantized model from {args.qmodel_path} with quantization config: {model.config.quantization_config}")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.config.name_or_path, use_fast=False
    )

    process_max_memory = group_gpu(model, accelerator.num_processes)
    if local_rank in process_max_memory:  # group main processor
        max_memory = process_max_memory[local_rank]
        device_map = infer_auto_device_map(model, max_memory, model._no_split_modules)
        dispatch_model(model, device_map)
        utils.cleanup_memory()
    accelerator.wait_for_everyone()

    group_main_ranks = list(process_max_memory.keys())
    logging.info(f"Group GPU Map: {process_max_memory}")

    if args.ppl_eval:
        for eval_dataset in args.eval_datasets:
            # Evaluating on dataset
            testloader = data_utils.get_loaders(
                eval_dataset, tokenizer, seqlen=2048, eval_mode=True  # type: ignore
            )
            logging.info(f"Evaluating {eval_dataset} PPL...")
            dataset_ppl, nlls = eval_utils.ppl_eval(
                model, testloader, args.ppl_eval_bsz, group_main_ranks
            )
            logging.info(f"{eval_dataset} PPL: {dataset_ppl}")
            utils.cleanup_memory(verbos=True)

    if not args.lm_eval:
        return

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
    )

    # manual setup for distributed eval
    # hflm dose not support pre-distributed model evaluation
    if accelerator.num_processes > 1:
        hflm.accelerator = accelerator
        hflm._rank = local_rank
        hflm._world_size = len(group_main_ranks)

    raw_results = {}
    results = {}
    for task_name in args.tasks:
        logging.info(f"Evaluating {task_name}...")
        result = simple_evaluate(hflm, tasks=[task_name], cache_requests=True)
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


def parser_gen(dist_rank=0):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qmodel_path",
        type=str,
        help="Model to load",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="path to save log",
    )

    # Quantization Arguments
    parser.add_argument(
        "--adj_layers",
        type=str,
        nargs="+",
        choices=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        help="Which layers to apply Hadamard adjustment. Override qmodel config if specified.",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        help="Asymmetric Activation quantization. Override qmodel config if specified.",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio. Override qmodel config if specified.",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio. Override qmodel config if specified.",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio. Override qmodel config if specified.",
    )

    # PPL Eval Arguments
    parser.add_argument(
        "--ppl_eval",
        action="store_true",
        help="Evaluate the model perplexity on the eval_dataset",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=["wikitext2", "c4"],
        help="Datasets for Evaluation (default: wikitext2)",
        choices=utils.supported_datasets,
    )
    parser.add_argument(
        "--ppl_eval_bsz",
        type=int,
        default=1,
        help="Batch-size for PPL evaluation (default:1)",
    )

    # LM Eval Arguments
    parser.add_argument(
        "--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "arc_challenge",
            "arc_easy",
            "hellaswag",
            "lambada_openai",
            "piqa",
            "winogrande",
        ],
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        type=str,
        default="auto:6",
        help="Batch size for evaluating with lm eval harness.",
    )

    args = parser.parse_args()

    # save current git hash
    try:
        import subprocess

        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        args.git_hash = git_hash
    except Exception as e:
        args.git_hash = "N/A"

    if args.log_path is None:
        args.log_path = os.path.dirname(args.qmodel_path)

    now_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    print(args, now_datetime)
    args.save_log_dir = os.path.join(args.log_path, f"{now_datetime}")
    os.makedirs(args.save_log_dir, exist_ok=True)
    utils.config_logging(
        os.path.join(args.save_log_dir, f"rank{dist_rank}.log"),
        dist_rank=dist_rank,
    )

    logging.info("Arguments: ")
    logging.info(pprint.pformat(vars(args)))
    logging.info("--" * 30)
    return args


if __name__ == "__main__":
    main()
