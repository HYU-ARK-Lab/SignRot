"""Deployment-kernel integration for the --use_kernel path.

Replaces the remaining fake-quant components with the real deployment modules
used by the e2e model (e2e/quantized_llama/modeling_llama.py):
- attention KV cache -> signrot.transformers.MultiLayerPagedKVCache4Bit
  (int4 pack + K Hadamard on update, flashinfer batch decode at generation)
- prefill attention  -> flash-attention 2 over the original fp16 KV
- RMSNorm            -> signrot.nn.RMSNorm

The linear layers themselves (act quant / int4 matmul / dequant / online
Hadamard) are handled by ActQuantWrapper.enable_kernel (quant_utils.py).
"""

import functools
import logging
import types

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import model_utils
import signrot.functional.kernel as signrot_kernel
import signrot.nn
from signrot.transformers import MultiLayerPagedKVCache4Bit


class PagedKVCacheAdapter(Cache):
    """Adapt the deployment MultiLayerPagedKVCache4Bit to the transformers 4.5x Cache API."""

    is_compileable = False

    def __init__(self, inner: MultiLayerPagedKVCache4Bit):
        super().__init__(layers=[])
        self.inner = inner

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx=0):
        return self.inner.get_seq_length(layer_idx)

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.inner.length + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        # rewind for repeated decode benchmarking; pages past max_length are overwritten
        self.inner.length = max_length

    def reset(self):
        self.inner.reset()


# The cache allocates whole pages, so a page size close to the sequence length
# wastes up to a full page per sequence (at 2048 a 2176-token run allocates 4096
# slots).  A smaller page also gives the decode kernel several pages per
# sequence to split its work across.
DEFAULT_PAGE_SIZE = 256


def build_paged_cache(model, batch_size, max_length, k_bits, page_size=None):
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    disable_quant = k_bits >= 16
    inner = MultiLayerPagedKVCache4Bit(
        batch_size=batch_size,
        page_size=min(page_size or DEFAULT_PAGE_SIZE, max_length),
        max_seq_len=max_length,
        device=model.get_input_embeddings().weight.device,
        n_layers=len(model_utils.get_layers(model)),
        num_heads=config.num_attention_heads,
        head_dim=head_dim,
        disable_quant=disable_quant,
        hadamard_dtype=None if disable_quant else torch.float16,
        group_size=config.num_attention_heads // config.num_key_value_heads,
    )
    return PagedKVCacheAdapter(inner)


def kernel_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = hidden_states.shape[1]

    if _shares_input(self.q_proj, self.k_proj, self.v_proj):
        # q/k/v read the same activation, so quantize it once (three identical
        # quantizations per layer is a third of all the quant launches)
        packed = self.q_proj.kernel_quantize(hidden_states)
        dtype = hidden_states.dtype
        query_states = self.q_proj.kernel_matmul(packed, dtype).view(hidden_shape)
        key_states = self.k_proj.kernel_matmul(packed, dtype).view(hidden_shape)
        value_states = self.v_proj.kernel_matmul(packed, dtype).view(hidden_shape)
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
    if hasattr(self, "q_norm"):  # Qwen3
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    fp16_rope = (
        cos.dtype == torch.float16
        and query_states.dtype == torch.float16
        and key_states.dtype == torch.float16
    )
    if fp16_rope:
        # one fused in-place kernel instead of the 10 elementwise ops
        # (mul/neg/cat/mul/add per tensor) apply_rotary_pos_emb issues
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        signrot_kernel.rope_qk(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

    # signrot kernels (KV quant/pack + flashinfer decode) are fp16-only
    query_states = query_states.to(torch.float16)
    key_states = key_states.to(torch.float16)
    value_states = value_states.to(torch.float16)

    if past_key_values is not None:
        cache_out = past_key_values.update(
            key_states, value_states, self.layer_idx, {"attention_mask": attention_mask}
        )
    else:
        cache_out = (key_states, value_states)

    if isinstance(cache_out, tuple):
        # prefill: the cache stores the Hadamard-rotated int4 KV and returns the
        # original fp16 KV for flash attention (deployment behavior)
        key_states, value_states = cache_out
        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, is_causal=True
        )
    else:
        # decode: flashinfer batch decode over the int4 paged KV cache
        # (the stub rotates q with the same per-head Hadamard as the cached K)
        attn_output = cache_out(query_states)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous().to(hidden_states.dtype)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


class GraphedDecoder:
    """Replay a single decode step from a captured CUDA graph.

    Batch-1 decode on this stack is host-bound: the forward issues ~1.5k tiny
    kernels whose launch cost exceeds their runtime, so the GPU idles most of
    the step.  Capturing the step once and replaying it removes the per-kernel
    launch cost entirely.

    Everything the step needs that changes between tokens lives in fixed
    buffers refreshed from outside the graph: the input token, the cache
    position (which drives RoPE), and the paged cache's index tensors.  The
    Python bookkeeping inside `MultiLayerPagedKVCache4Bit.update` does not run
    on replay, so the sequence length is advanced here instead.

    Capture writes a throwaway token into the cache and then rewinds the length,
    so the slot is overwritten by the first real step.
    """

    def __init__(self, model, cache, batch_size):
        self.model = model
        self.cache = cache
        self.inner = getattr(cache, "inner", cache)
        device = model.get_input_embeddings().weight.device
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None
        self.logits = None

    def _forward(self):
        return self.model(
            self.input_ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.cache_position,
        ).logits

    def _prepare_step(self):
        """Advance the host-side length and refresh the index buffers."""
        self.cache_position.fill_(self.inner.length)
        self.inner.length += 1
        self.inner.refresh_specs()

    def capture(self, warmup=3):
        start_length = self.inner.length
        self.inner.freeze_specs(True)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self.inner.length = start_length
                    self._prepare_step()
                    self._forward()
            torch.cuda.current_stream().wait_stream(stream)

            self.inner.length = start_length
            self._prepare_step()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self._forward()
        finally:
            self.inner.length = start_length
            self.inner.freeze_specs(False)
        return self

    def step(self, input_ids):
        """Run one decode step; returns the logits buffer (overwritten next step)."""
        if self.graph is None:
            raise RuntimeError("capture() must run before step()")
        self.input_ids.copy_(input_ids)
        self.inner.freeze_specs(True)
        self._prepare_step()
        self.graph.replay()
        self.inner.freeze_specs(False)
        return self.logits


def _graph_matches_eager(model, cache, decoder, batch_size, probes=2, tol=1e-3):
    """Check a captured graph reproduces the eager step on fresh inputs.

    Capture records a fixed sequence of kernels, so any host-side branching on
    tensor values -- MoE expert routing is the obvious case -- gets frozen at
    whatever the capture inputs selected.  That failure is silent, so the graph
    is checked against eager on tokens it has not seen.
    """
    inner = decoder.inner
    device = decoder.input_ids.device
    vocab = model.get_input_embeddings().weight.shape[0]
    start = inner.length
    try:
        for _ in range(probes):
            # CPU sampling: the CUDA RNG may be unusable after a failed capture
            token = torch.randint(0, vocab, (batch_size, 1), dtype=torch.long).to(device)
            inner.length = start
            eager = model(token, past_key_values=cache, use_cache=True).logits.float()
            inner.length = start
            graphed = decoder.step(token).float()
            rel = (graphed - eager).abs().max() / eager.abs().max().clamp(min=1e-6)
            if not torch.isfinite(rel) or rel.item() > tol:
                logging.warning(
                    f"CUDA graph output diverges from eager (relative {rel.item():.3e}); "
                    f"the decode step is not capture-safe"
                )
                return False
    finally:
        inner.length = start
    return True


def try_graph_decoder(model, cache, batch_size, warmup=3, validate=True):
    """Capture a decode CUDA graph, or return None if the stack refuses capture."""
    if not torch.cuda.is_available():
        return None
    try:
        decoder = GraphedDecoder(model, cache, batch_size).capture(warmup=warmup)
    except Exception as exc:  # capture is best-effort; the eager path still works
        logging.warning(f"CUDA graph capture failed, falling back to eager decode: {exc}")
        torch.cuda.synchronize()
        return None
    if validate and not _graph_matches_eager(model, cache, decoder, batch_size):
        logging.warning("Falling back to eager decode")
        return None
    logging.info("Captured a CUDA graph for the decode step")
    return decoder


def _shares_input(first, *others):
    """Whether sibling projections can share one quantized activation."""
    shares = getattr(first, "shares_kernel_input_with", None)
    return shares is not None and all(shares(o) for o in others)


def build_moe_block(mlp):
    """Convert an enable_kernel'd Qwen3MoeSparseMoeBlock into a
    signrot.nn.SignRotMoEBlock over stacked expert weights.

    The HF block runs one ActQuantWrapper kernel chain per expert projection
    behind a host-synced Python loop; the stacked block replaces that with
    GPU-side routing, grouped int4 GEMMs and the per-expert had_adj signs
    fused into the down-proj quantizer.  Returns None (leaving the HF block
    in place) when the experts are not uniformly kernel-enabled.
    """
    gates = [e.gate_proj for e in mlp.experts]
    ups = [e.up_proj for e in mlp.experts]
    downs = [e.down_proj for e in mlp.experts]
    wrappers = gates + ups + downs

    if not all(getattr(w, "use_kernel", False) for w in wrappers):
        logging.warning("MoE experts are not all kernel-enabled; keeping the HF expert loop")
        return None
    quantizer = gates[0].kernel_quantizer
    assert all(
        type(w.kernel_quantizer) is type(quantizer)
        and w.kernel_quantizer.clip_ratio == quantizer.clip_ratio
        for w in wrappers
    ), "MoE stacking expects one activation quantizer config across all experts"
    assert all(w.online_full_had for w in downs), \
        "expert down_proj must run the online Hadamard on the kernel path"
    assert all(w.fp32_had == downs[0].fp32_had for w in downs)

    asym = isinstance(quantizer, signrot.nn.AsymQuantizer)
    w_gate, s_gate, _ = signrot.nn.SignRotMoEBlock.stack_experts(
        [w.kernel_module for w in gates], need_wsum=False)
    w_up, s_up, _ = signrot.nn.SignRotMoEBlock.stack_experts(
        [w.kernel_module for w in ups], need_wsum=False)
    w_down, s_down, _ = signrot.nn.SignRotMoEBlock.stack_experts(
        [w.kernel_module for w in downs], need_wsum=False)

    sign_down = None
    if any(w.kernel_hadamard.had_adj_sign is not None for w in downs):
        inter = w_gate.shape[1]
        sign_down = torch.stack([
            w.kernel_hadamard.had_adj_sign.to(w_gate.device)
            if w.kernel_hadamard.had_adj_sign is not None
            # never-routed experts kept the identity adjustment
            else torch.ones(inter, dtype=torch.int8, device=w_gate.device)
            for w in downs
        ]).contiguous()

    return signrot.nn.SignRotMoEBlock(
        mlp.gate, top_k=mlp.top_k, norm_topk_prob=mlp.norm_topk_prob,
        w_gate=w_gate, s_gate=s_gate, w_up=w_up, s_up=s_up,
        w_down=w_down, s_down=s_down, sign_down=sign_down,
        asym=asym, clip_ratio=quantizer.clip_ratio, fp32_had=downs[0].fp32_had,
    )


def kernel_mlp_forward(self, x):
    """SwiGLU MLP with the silu/multiply pair fused into one kernel."""
    if _shares_input(self.gate_proj, self.up_proj):
        packed = self.gate_proj.kernel_quantize(x)
        gate = self.gate_proj.kernel_matmul(packed, x.dtype)
        up = self.up_proj.kernel_matmul(packed, x.dtype)
    else:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
    if gate.dtype == torch.float16:
        return self.down_proj(signrot_kernel.swiglu(gate, up))
    return self.down_proj(self.act_fn(gate) * up)


def patch_kernel_model(model, args):
    """Switch attention, KV cache and RMSNorm of a fake-quant model to the
    deployment kernel modules. Must run after enable_kernel_modules and before
    dispatch_model. The paged KV cache lives on a single device, so the model
    must not be split across GPUs."""
    moe_blocks = 0
    for layer in model_utils.get_layers(model):
        layer.self_attn.forward = types.MethodType(kernel_attention_forward, layer.self_attn)
        mlp = getattr(layer, "mlp", None)
        if isinstance(getattr(mlp, "act_fn", None), torch.nn.SiLU):
            mlp.forward = types.MethodType(kernel_mlp_forward, mlp)
        elif model_utils.is_moe_layer(layer) and not getattr(args, "no_moe_kernel", False):
            # replacing the mlp drops the last reference to the per-expert
            # modules, so each layer's duplicated weights are freed as we go
            stacked = build_moe_block(mlp)
            if stacked is not None:
                layer.mlp = stacked
                moe_blocks += 1

    replaced_norms = 0
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, model_utils.RMSN):
                setattr(
                    parent,
                    child_name,
                    signrot.nn.RMSNorm(child.mean_dim, eps=child.eps),
                )
                replaced_norms += 1

    # keep the mask path in FA2 semantics: attention_mask stays 2D/None and our
    # attention forward + the paged cache consume it directly
    model.config._attn_implementation = "flash_attention_2"

    model._expected_max_length = None
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward_with_paged_cache(
        input_ids=None, *fargs, past_key_values=None, use_cache=None, **fkwargs
    ):
        use_cache = use_cache if use_cache is not None else model.config.use_cache
        if use_cache and (past_key_values is None or isinstance(past_key_values, DynamicCache)):
            max_length = max(model._expected_max_length or 0, input_ids.shape[1])
            model._expected_max_length = None
            past_key_values = build_paged_cache(
                model, input_ids.shape[0], max_length, args.k_bits
            )
        return original_forward(
            input_ids, *fargs, past_key_values=past_key_values, use_cache=use_cache, **fkwargs
        )

    model.forward = forward_with_paged_cache
    logging.info(
        f"Patched attention with kernel KV cache in {len(model_utils.get_layers(model))} layers, "
        f"replaced {replaced_norms} RMSNorms with signrot.nn.RMSNorm"
        + (f", stacked {moe_blocks} MoE blocks into SignRotMoEBlock" if moe_blocks else "")
    )
