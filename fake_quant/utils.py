# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging
from termcolor import colored

from accelerate import PartialState
from accelerate.utils import gather_object

supported_models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    # "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    # "Qwen/Qwen2-7B",
    "Qwen/Qwen3-30B-A3B",
    # "facebook/opt-125m",
]
supported_datasets = ["wikitext2", "ptb", "c4"]
supported_adjust_calibration_datasets = [
    "wikitext2",
    "ptb",
    "c4",
    # "handwritten_prompt",
]

# handwritten_prompt = """This is a diverse prompt that contains:
#     - Fiction: "Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ..."
#     - News: "The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ..."
#     - Code: `public static void main(String[] args) {\n System.out.println("Hello world!);\n}`
#     - Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
#     - Facts: "The capital of Egypt is Cairo. It is the largest city in the region and is home to..."
#     """

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = PartialState().device


def get_down_proj_groupsize(model, groupsize):

    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f"(Act.) Groupsiz = Down_proj Groupsize: {groupsize}")
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert (
        groupsize * group_num == model.config.hidden_size
    ), "Invalid groupsize!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert (
        down_proj_groupsize * group_num == model.config.intermediate_size
    ), "Invalid groupsize for down_proj!"
    logging.info(
        f"(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}"
    )
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file, dist_rank=0):

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )
    date_fmt = "%Y-%m-%d:%H:%M:%S"

    handlers = []
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt=date_fmt))
        handlers.append(console_handler)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
    handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)


def parser_gen(dist_rank=0):
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model to load;",
        choices=supported_models,
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )
    parser.add_argument("--hf_token", type=str, default=None)

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    parser.add_argument(
        "--adj_layers",
        type=str,
        nargs="*",
        choices=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        default=[],
        help="Which layers to apply Hadamard adjustment",
    )
    parser.add_argument(
        "--adj_layer_ids",
        type=str,
        nargs="+",
        default="all",
        help="Which layers to apply Hadamard adjustment. default to all layers.",
    )
    parser.add_argument(
        "--adj_cal_dataset",
        type=str,
        default=None,
        help="calibration data for Hadamard adjustment.",
        choices=supported_adjust_calibration_datasets,
    )
    parser.add_argument(
        "--adj_cal_dataset_seed", type=int, default=0, help="Random Seed for hadamard adjustment"
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--a_extend_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extend range for activation quantization. new_max = maximum(max, 0), new_min = minimum(min, 0)",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_quantizer",
        type=str,
        default="gptq",
        choices=["gptq", "gptaq", "rtn"],
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--w_extend_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extend range for weignt quantization. new_max = maximum(max, 0), new_min = minimum(min, 0)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--cal_batch_size",
        type=int,
        default=8,
        help="Batch size for calibration forward passes during GPTQ.",
    )
    parser.add_argument(
        "--cal_dataset",
        type=str,
        default="wikitext2",
        help="calibration data samples for GPTQ.",
        choices=supported_datasets,
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )
    parser.add_argument(
        "--gptq_streams",
        type=int,
        default=1,
        help="Number of CUDA streams for GPTQ fasterquant. Use 1 to disable.",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )
    parser.add_argument(
        "--use_kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Run evaluation through the real deployment kernels instead of fake quantization:
                        signrot linear kernels (act quant/int4 matmul/dequant), OnlineHadamard (+had_adj),
                        signrot RMSNorm and the int4 paged KV cache with flashinfer decode attention.""",
    )
    parser.add_argument(
        "--no_moe_kernel",
        action="store_true",
        help="""With --use_kernel, keep the HF per-expert MoE loop instead of stacking
                        experts into the grouped-GEMM SignRotMoEBlock (debug escape hatch).""",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=str, default="-1")
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--v_extend_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extend range for v-cache quantization. new_max = maximum(max, 0), new_min = minimum(min, 0)",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=str, default="-1")
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )
    parser.add_argument(
        "--k_extend_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extend range for k-cache quantization. new_max = maximum(max, 0), new_min = minimum(min, 0)",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--save_qmodel",
        nargs="?",
        const=True,
        default=False,
        help="Save the quantized model to the save_dir",
    )
    parser.add_argument(
        "--load_qmodel",
        nargs="?",
        const=True,
        default=False,
        help="Load the quantized model",
    )
    parser.add_argument(
        "--packed_safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save/Load the quantized model using packed int4 safetensors format",
    )

    parser.add_argument(
        "--cache_rotated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache rotated model to/from experiments/<model>/<rotate_mode>_<rotation_seed>.pt",
    )

    # Experiments Arguments
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="The path to save experiment data, "
        "including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].",
    )
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )
    parser.add_argument(
        "--log_ma_position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log the max-abs channel indices during eval",
    )

    # distribute_mode
    parser.add_argument(
        "--distribute_mode",
        type=str,
        default="greedy",
        choices=["greedy", "balanced"],
        help="Distribute the model on multiple GPUs for evaluation.",
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
        choices=supported_datasets,
    )
    parser.add_argument(
        "--ppl_eval_bsz",
        type=int,
        default=1,
        help="Batch-size for PPL evaluation (default:1)",
    )

    # Runtime Benchmark Arguments
    parser.add_argument(
        "--runtime_bench",
        action="store_true",
        help="Measure prefill/decode/e2e generation runtime and exit (skips ppl/lm eval).",
    )
    parser.add_argument(
        "--bench_bsz", type=int, nargs="+", default=[1], help="Batch sizes to benchmark."
    )
    parser.add_argument(
        "--bench_prefill_len",
        type=int,
        nargs="+",
        default=[2048],
        help="Prefill sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--bench_decode_steps",
        type=int,
        default=128,
        help="Number of decode steps per generation.",
    )
    parser.add_argument("--bench_repeats", type=int, default=5)
    parser.add_argument("--bench_warmup", type=int, default=1)
    parser.add_argument(
        "--bench_soak_s",
        type=float,
        default=0.0,
        help="Keep warming up on the measured workload until the GPU has been busy "
        "this long (in addition to --bench_warmup iterations). Consumer cards boost "
        "hard from cold and settle as they heat, so without a soak the first samples "
        "of a run are measured at a different clock than the last; the model load is "
        "CPU-bound, so every process starts cold again.",
    )
    parser.add_argument(
        "--bench_log_clocks",
        action="store_true",
        help="Sample SM clock / temperature / power into gpu_telemetry.csv next to the "
        "run log, so measured drift can be attributed to (or cleared of) throttling.",
    )
    parser.add_argument(
        "--bench_no_cuda_graph",
        action="store_true",
        help="Do not replay decode from a CUDA graph (the eager path is host-bound, "
        "so this measures launch overhead rather than the kernels).",
    )
    parser.add_argument(
        "--bench_skip_oom",
        action="store_true",
        help="Skip configurations that run out of memory instead of aborting the sweep.",
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
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
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

    quant_type = f"w{args.w_bits}a{args.a_bits}kv{args.k_bits}"

    now_datetime = datetime.now().strftime("%Y%m%d_%H%M")

    if args.save_name is None:
        args.save_name = now_datetime

    setattr(
        args,
        "save_path",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "experiments",
            args.model,
            args.save_name,
        ),
    )

    args.save_log_dir = os.path.join(args.save_path, f"{now_datetime}")
    os.makedirs(args.save_log_dir, exist_ok=True)
    config_logging(
        os.path.join(args.save_log_dir, f"rank{dist_rank}.log"),
        dist_rank=dist_rank,
    )

    if args.load_qmodel == True:
        args.load_qmodel = os.path.join(args.save_path, "qmodel")
    if args.save_qmodel == True:
        args.save_qmodel = os.path.join(args.save_path, "qmodel")

    assert (
        args.a_groupsize == args.w_groupsize
    ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope == False, "Pre-RoPE quantization is not supported yet!"

    if args.runtime_bench:
        # --load_qmodel is supported here (main.py loads it inside the
        # runtime_bench branch): the kernel path is configured from the layer
        # topology + weight quantizers, not from the weight values, so a saved
        # qmodel benchmarks the same kernels as a freshly quantized one while
        # skipping rotation + RTN.  Saving on top of a load is still nonsense.
        assert not (args.load_qmodel and args.save_qmodel), (
            "--load_qmodel and --save_qmodel are mutually exclusive"
        )

    if args.use_kernel:
        assert args.w_bits == 4 and args.a_bits == 4, "--use_kernel requires W4A4 quantization!"
        assert not args.w_asym, "--use_kernel only supports symmetric weight quantization!"
        assert args.w_groupsize == -1 and args.a_groupsize == -1, "--use_kernel does not support groupwise quantization!"
        assert not args.a_extend_zero, "--use_kernel does not support a_extend_zero!"
        assert not args.int8_down_proj, "--use_kernel does not support int8_down_proj!"
        assert args.k_bits in (4, 16) and args.v_bits == args.k_bits, (
            "--use_kernel supports KV4 (int4 paged cache) or KV16 (fp16 cache) only, "
            "with k_bits == v_bits"
        )

    if args.model == "facebook/opt-125m" or args.model == "facebook/opt-1.3b":
        logging.warning("Warning: OPT-125M/1.3B is only for debugging purposes!!")

    logging.info("Arguments: ")
    logging.info(pprint.pformat(vars(args)))
    logging.info("--" * 30)
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


import contextlib, functools

@contextlib.contextmanager
def detect_initial_nan(model):
    # hook initial NaN occurred points
    def nan_hook(module, inputs, outputs, name="N/A"):
        # find tensor from outputs
        if isinstance(outputs, torch.Tensor):
            output = outputs
        elif isinstance(outputs, (list, tuple)):
            output = None
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    output = out
                    break
            if output is None:
                return
        else:
            return

        if torch.isnan(output).any():
            logging.error(f"NaN detected in {name}")
            raise ValueError(f"NaN detected in {name}")
    nan_hooks = []
    for name, module in model.named_modules():
        hook = functools.partial(nan_hook, name=name)
        nan_hooks.append(module.register_forward_hook(hook))

    try:
        yield
    finally:
        for hook in nan_hooks:
            hook.remove()
