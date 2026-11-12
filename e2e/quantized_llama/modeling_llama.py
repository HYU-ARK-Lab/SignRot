# Modified from Hugging Face Transformers (https://github.com/huggingface/transformers) via QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import functools
import signrot
import signrot.transformers
import torch
from transformers import LlamaConfig, GenerationMixin
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import Cache, DynamicCache


ALL_LAYERNORM_LAYERS.append(signrot.nn.RMSNorm)

class SignrotLlamaConfig(LlamaConfig):
    model_type = "llama_signrot"

class SignrotFP16LlamaAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_kwargs=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # kv_seq_len = key_states.shape[1]
        # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2
        )

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = _flash_attention_forward(
                query_states, 
                key_states, 
                value_states, 
                query_length=q_len, 
                attention_mask=attention_mask,
                is_causal=True,
            )
        else:
            attn_output = cache_out(query_states)

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class SignrotLlamaAttention(SignrotFP16LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.get("config", {})
        quant_config = getattr(config, "quantization_config", {})
        a_clip_ratio = quant_config.get("a_clip_ratio", 1.0)
        a_asym = quant_config.get("a_asym", False)
        Quantizer = signrot.nn.AsymQuantizer if a_asym else signrot.nn.Quantizer

        self.quantizer = Quantizer(clip_ratio=a_clip_ratio)
        self.q_proj = signrot.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = signrot.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = signrot.nn.Linear4bit.from_float(self.v_proj)
        o_proj = signrot.nn.Linear4bit.from_float(self.o_proj)
        self.o_proj_hadamard = signrot.nn.OnlineHadamard(self.num_heads, o_proj)  # tie to hadamard
        self.o_proj = torch.nn.Sequential(Quantizer(clip_ratio=a_clip_ratio), o_proj)

class SignrotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.get("config", {})
        quant_config = getattr(config, "quantization_config", {})
        a_clip_ratio = quant_config.get("a_clip_ratio", 1.0)
        a_asym = quant_config.get("a_asym", False)
        Quantizer = signrot.nn.AsymQuantizer if a_asym else signrot.nn.Quantizer

        had_adj = "down_proj" in quant_config.get("adj_layers", [])
        self.quantizer = Quantizer(clip_ratio=a_clip_ratio)
        self.up_proj = signrot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = signrot.nn.Linear4bit.from_float(self.gate_proj)
        down_proj = signrot.nn.Linear4bit.from_float(self.down_proj)
        self.down_proj = torch.nn.Sequential(
            signrot.nn.OnlineHadamard(self.intermediate_size, down_proj, had_adj=had_adj),  # tie to hadamard
            Quantizer(clip_ratio=a_clip_ratio),
            down_proj
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class SignrotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = SignrotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None

        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_attention_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // self.config.num_attention_heads
        group_size = num_heads // self.config.num_key_value_heads
        disable_quant = self.cache_dtype == "float16"

        return signrot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16,
            group_size=group_size,
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None or isinstance(past_key_values, DynamicCache):
            max_length = max(self._expected_max_length or 0, input_ids.shape[1])
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=min(2048, max_length),  # For now working with single page per batch.
                max_length=max_length)
            past_key_values = signrot.transformers.HFCacheAdapter(past_key_values)
        elif not isinstance(past_key_values, Cache):
            past_key_values = signrot.transformers.HFCacheAdapter(past_key_values)

        kwargs.pop("use_cache", None)
        out = super().forward(input_ids, *args, past_key_values=past_key_values, use_cache=True, **kwargs)
        return out
    


class SignrotLlamaForCausalLM(SignrotFP16LlamaForCausalLM, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        self.model.norm = signrot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = SignrotLlamaAttention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = signrot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = signrot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = SignrotLlamaMLP(config=config)
        self.cache_dtype = "int4"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        config=None,
        **kwargs
    ):
        import os
        from transformers.modeling_utils import no_init_weights
        from fake_quant.realquant_utils import Safetensors
        import tqdm
        
        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config._attn_implementation = "flash_attention_2"

        # TODO: assert unsupported args
        # ex) k_asym == False or v_asym == False, int8_down_proj == True

        quantization_config = kwargs.pop("quantization_config", None)
        exist_quantization_config = getattr(config, "quantization_config", {})
        if quantization_config is not None:
            exist_quantization_config.update(quantization_config)
            config.quantization_config = exist_quantization_config

        with no_init_weights():
            model = cls(config)

        if not os.path.isdir(pretrained_model_name_or_path):
            raise ValueError(f"Only local loading is supported for quantized models. Got {pretrained_model_name_or_path}")

        state_dict = {}
        device_map = kwargs.get("device_map", None)
        with Safetensors(pretrained_model_name_or_path, device_map) as f:
            for k in tqdm.tqdm(f.keys(), desc="Loading quantized weights"):
                tensor = f.get_tensor(k)
                if k.startswith("quantizer."):
                    k = k.replace("quantizer.", "").replace(".scale", ".weight_scales")
                k = (k.replace("o_proj.weight", "o_proj.1.weight")
                    .replace("o_proj.weight_scales", "o_proj.1.weight_scales")
                    .replace("down_proj.weight", "down_proj.2.weight")
                    .replace("down_proj.weight_scales", "down_proj.2.weight_scales")
                    .replace("down_proj.had_adj_sign", "down_proj.0.had_adj_sign"))
                state_dict[k] = tensor

        model.load_state_dict(state_dict, strict = False)

        torch.set_default_dtype(dtype_old)
        return model
