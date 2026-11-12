import os
import logging
from typing import Generator, Any, TypeAlias
from contextlib import contextmanager, ExitStack
import json
from types import SimpleNamespace

from accelerate import PartialState
from safetensors import safe_open
import tqdm
import torch

import quant_utils


## save weight in uint8 with safetensors
def save_quantized_weights_with_safetensors(
    state_dict,
    quantizers: dict[str, quant_utils.WeightQuantizer],
    config,
    args,
):
    import json
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards
    import signrot

    metadata = {}
    max_shard_size = "5GB"
    save_dir = args.save_qmodel

    # fmt: off
    quantization_config = {
        "quant_method": "SignRot",
        "w_quantizer": args.w_quantizer,
        "w_bits": args.w_bits, "w_asym": args.w_asym, "w_groupsize": args.w_groupsize, "w_clip": args.w_clip, "w_perchannel": True,
        "a_bits": args.a_bits, "a_asym": args.a_asym, "a_groupsize": args.a_groupsize, "a_clip_ratio": args.a_clip_ratio,
        "k_bits": args.k_bits, "k_asym": args.k_asym, "k_groupsize": args.k_groupsize, "k_clip_ratio": args.k_clip_ratio,
        "v_bits": args.v_bits, "v_asym": args.v_asym, "v_groupsize": args.v_groupsize, "v_clip_ratio": args.v_clip_ratio,
        
        "format": "packed_int4",
        
        "rotate": args.rotate,
        "rotate_mode": args.rotate_mode,
        "fp32_had": args.fp32_had,
        "adj_layers": args.adj_layers,
        
        "int8_down_proj": args.int8_down_proj,
    }
    # fmt: on
    metadata["quantization_config"] = json.dumps(quantization_config)

    w_sym = not args.w_asym
    quantize_weight = quant_utils.sym_quant if w_sym else quant_utils.asym_quant

    for name, param in tqdm.tqdm(state_dict.items(), desc="Packing weights"):
        if "module" in name:
            continue
        if name.endswith(".weight") or name.endswith(".bias"):
            layer_name = name.rsplit(".", 1)[0]
        else:
            layer_name = name

        param_device = param.device
        quantizer_key = f"{layer_name}.module"
        quantizer = quantizers[quantizer_key] if quantizer_key in quantizers else None

        if "weight" in name and quantizer is not None:
            quant_kwargs = {
                "scale": quantizer.scale.to(param_device),
                "maxq": quantizer.maxq.to(param_device),
            }
            if not w_sym:
                quant_kwargs["zero"] = quantizer.zero.to(param_device)

            param_quant_int8 = quantize_weight(param, **quant_kwargs)[0].to(
                torch.int8
            )
            param_qpacked_int4 = quant_utils.pack_i4(param_quant_int8).cpu()
            state_dict[name] = param_qpacked_int4.contiguous()
        else:
            state_dict[name] = param.to(torch.half).contiguous()

    for quantizer_key, quantizer in quantizers.items():
        layer_name = quantizer_key.rsplit(".", 1)[0]
        state_dict[f"quantizer.{layer_name}.scale"] = quantizer.scale.contiguous()
        if hasattr(quantizer, "zero") and quantizer.zero is not None:
            state_dict[f"quantizer.{layer_name}.zero"] = quantizer.zero.contiguous()
        if hasattr(quantizer, "maxq") and quantizer.maxq is not None:
            state_dict[f"quantizer.{layer_name}.maxq"] = quantizer.maxq.contiguous()

    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
    )

    shards = {}
    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard_state_dict = {}
        for tensor_name in tensor_names:
            shard_state_dict[tensor_name] = state_dict[tensor_name]
        shards[filename] = shard_state_dict

    # Save shards
    first_shard = True
    for shard_file, shard_state_dict in shards.items():
        shard_path = os.path.join(save_dir, shard_file)

        # Only add metadata to the first file
        if first_shard:
            save_file(shard_state_dict, shard_path, metadata=metadata)
            first_shard = False
        else:
            save_file(shard_state_dict, shard_path)
        print(f"Saved {shard_file}")

    # Save index
    if state_dict_split.is_sharded:
        index = {
            "metadata": (
                state_dict_split.metadata
                if hasattr(state_dict_split, "metadata")
                else {}
            ),
            "weight_map": state_dict_split.tensor_to_filename,
        }
        index_path = os.path.join(save_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Saved index to {index_path}")

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {**config, "quantization_config": quantization_config},
            f,
            indent=2,
        )

    logging.info(f"saved weights at {save_dir}")


DeviceTypeLike: TypeAlias = str | int | torch.device | dict | None


def to_device_map(device_map: DeviceTypeLike):
    if device_map is None:
        return {"": torch.device("cpu")}
    if isinstance(device_map, (str, int, torch.device)):
        return {"": torch.device(device_map)}
    if isinstance(device_map, dict):
        return {k: torch.device(v) for k, v in device_map.items()}
    raise ValueError(f"Invalid device_map type: {type(device_map)}")


def is_module_prefix(key: str, module_prefix: str):
    return (
        module_prefix == ""
        or key == module_prefix
        or key.startswith(module_prefix + ".")
    )


def get_device_from_map(key, device_map: DeviceTypeLike = "cpu"):
    # device map can have format like {'': 'cpu'} or {'layer1': 'cuda:0', 'layer2': 'cuda:1'}
    # but key can be 'layer1.weight' or 'layer2.bias'
    if device_map is None:
        return torch.device("cpu")
    if isinstance(device_map, (str, int, torch.device)):
        return torch.device(device_map)

    assert isinstance(device_map, dict)
    if "" in device_map:
        return torch.device(device_map[""])
    for prefix in device_map.keys():
        if is_module_prefix(key, prefix):
            return torch.device(device_map[prefix])

    # fallback to cpu
    return torch.device("cpu")


class Safetensors:
    def __init__(self, load_path: str, device_map: DeviceTypeLike = "cpu"):
        self.load_path = load_path
        self.device_map = to_device_map(device_map)

        index_path = os.path.join(self.load_path, "model.safetensors.index.json")
        self._metadata = {}
        self._weight_map = {}
        if os.path.exists(index_path):
            # Sharded model
            with open(index_path, "r") as f:
                index = json.load(f)
                self._metadata = index.get("metadata", {})
                self._weight_map = index["weight_map"]

            for filename in set(self._weight_map.values()):
                shard_path = os.path.join(load_path, filename)
        else:
            safetensor_path = os.path.join(load_path, "model.safetensors")
            with safe_open(safetensor_path, framework="pt") as f:
                self._metadata = f.metadata()
                for key in f.keys():
                    self._weight_map[key] = "model.safetensors"

        self.weight_map = {}
        self.safetensors: dict[tuple[str, torch.device], safe_open] = {}
        for prefix, device in self.device_map.items():
            for key, file_path in self._weight_map.items():
                if not is_module_prefix(key, prefix):
                    continue

                safetensor_key = (file_path, device)
                if safetensor_key not in self.safetensors:
                    safetensor = safe_open(
                        os.path.join(self.load_path, file_path),
                        framework="pt",
                        device=str(device),
                    )
                    self.safetensors[safetensor_key] = safetensor
                self.weight_map[key] = safetensor_key

        # fallback for keys not matched in device_map
        for key, file_path in self._weight_map.items():
            if key not in self.weight_map:
                safetensor_key = (file_path, torch.device("cpu"))
                if safetensor_key not in self.safetensors:
                    safetensor = safe_open(
                        os.path.join(self.load_path, file_path),
                        framework="pt",
                        device="cpu",
                    )
                    self.safetensors[safetensor_key] = safetensor
                self.weight_map[key] = safetensor_key

    def keys(self):
        return self.weight_map.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Key {key} not found in safetensors.")
        f = self.safetensors[self.weight_map[key]]
        return f.get_tensor(key)

    def __enter__(self):
        self.stack = ExitStack()
        for f in self.safetensors.values():
            self.stack.enter_context(f)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()


class Config(SimpleNamespace):
    def __init__(self, entries):
        super().__init__(**entries)

    def get[T](self, key, default: T = None):  # type: ignore
        return getattr(self, key, default)


@torch.no_grad()
def load_quantized_weights_with_safetensors(
    model, load_path, dequantize=False, device_map=None, dtype=torch.float16
):
    config = {}
    config_path = os.path.join(load_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    quant_config = Config(config.get("quantization_config", {}))
    logging.info(f"Load quantization config: {quant_config}")

    if device_map is None:
        if hasattr(model, "hf_device_map"):
            device_map = model.hf_device_map
    device_map = to_device_map(device_map)

    # add quantizers. to device_map
    for key in [*device_map.keys()]:
        if is_module_prefix(key, "model.layers"):
            device_map["quantizer." + key] = device_map[key]

    state_dict = {}
    quantizers = {}
    w_bits = quant_config.get("w_bits", 4)
    w_perchannel = quant_config.get("w_perchannel", True)
    w_asym = quant_config.get("w_asym", False)
    w_clip = quant_config.get("w_clip", False)

    # unpack int4 weights and load into model
    with Safetensors(load_path, device_map) as safetensors:
        state_dict_keys = []
        for k in safetensors.keys():
            if not k.startswith("quantizer."):
                state_dict_keys.append(k)
                continue

            # form of quantizer.{layer_name}.scale
            _, *layer_atoms, param_type = k.split(".")
            layer_name = ".".join(layer_atoms)
            v = safetensors.get_tensor(k)

            if layer_name not in quantizers:
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    bits=w_bits,
                    perchannel=w_perchannel,
                    sym=not w_asym,
                    mse=w_clip,
                )
                quantizers[layer_name] = quantizer

            if param_type == "scale":
                quantizers[layer_name].scale = v
            elif param_type == "zero":
                quantizers[layer_name].zero = v
            elif param_type == "maxq":
                quantizers[layer_name].maxq = v

        tqdm_prefix = f"[rank {PartialState().process_index}] Unpacking{" and dequantize" if dequantize else ""} weights"
        for k in (pbar := tqdm.tqdm(state_dict_keys, desc=tqdm_prefix)):
            pbar.set_description(tqdm_prefix + f" {k}")
            v = safetensors.get_tensor(k)
            if k.endswith(".weight"):
                layer_name = k.rsplit(".", 1)[0]
                if layer_name in quantizers:
                    quantizer = quantizers[layer_name]
                    v = quant_utils.unpack_i4(v, torch.int8)
                    if dequantize:
                        v = quantizer.dequantize(v)
            
            state_dict[k] = v.to(device=get_device_from_map(k, device_map), dtype=dtype)

    return state_dict, quantizers
