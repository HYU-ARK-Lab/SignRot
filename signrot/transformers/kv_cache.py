# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Dict, Any
import math
import torch
from .. import _CUDA
import functools
from fast_hadamard_transform import hadamard_transform
from signrot.functional.quantization import get_minq_maxq
import signrot.functional.kernel as signrot_kernel

@torch.jit.script
def asym_quantize_and_pack_i4(
    x: torch.Tensor,
    clip_factor_a_max: torch.Tensor = torch.tensor(1.0),
    clip_factor_a_min: torch.Tensor = torch.tensor(1.0),
    lac: bool = False,
    quantize: bool = True,
):
    minq, maxq = get_minq_maxq(bits=4, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)

    if lac:
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        xmax = xmax * clip_factor_a_max
        xmin = xmin * clip_factor_a_min

        xmin_zero = torch.eq(xmin, 0.0)
        xmax_zero = torch.eq(xmax, 0.0)
        zero_mask = xmin_zero * xmax_zero

        neg_one = torch.full_like(xmin, -1.0)
        pos_one = torch.full_like(xmax, 1.0)

        xmin = torch.where(zero_mask, neg_one, xmin)
        xmax = torch.where(zero_mask, pos_one, xmax)

        scale = (xmax - xmin) / maxq
        zero = torch.round((-1.0 * xmin) / scale)
        q = torch.clamp((x / scale).round() + zero, 0, maxq)

        if not quantize:
            return scale * (q - zero), scale, zero

    else:
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = -xmin
        q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

        if not quantize:
            return scale * (q - zero), scale, zero

    # pack int4
    q = q.to(dtype=torch.uint8)
    q = q[..., 0::2] | (q[..., 1::2] << 4)
    return q, scale, zero

def unpack_i4_and_asym_dequantize(q, scale, zero, lac: bool = False):
    #unpack int4
    assert q.dtype == torch.uint8
    q = torch.stack((q & 0x0f, (q >> 4) & 0x0f), dim=-1).view(*q.shape[:-1], q.shape[-1] * 2)

    if lac:
        return scale * (q - zero)
    else:
        return q * scale - zero

def matmul_had_cuda(X, dtype):
    n = X.shape[-1]
    input = hadamard_transform(X.to(dtype).contiguous(), scale=1/math.sqrt(n))
    return input.to(X.dtype).view(X.shape) 


def init_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _CUDA.init_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _CUDA.append_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)

def batch_decode_i4(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx, tmp=None, num_chunks=1):
    return _CUDA.batch_decode_i4(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx, tmp, num_chunks)


def init_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _CUDA.init_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _CUDA.append_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)

def batch_decode_f16(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _CUDA.batch_decode_f16(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)


class _AttentionStub(object):
    def __init__(self, cache_page_size, device, n_layers, disable_quant, hadamard_dtype):
        self.cache_page_size = cache_page_size
        self.n_layers = n_layers
        self.disable_quant = disable_quant
        self.hadamard_dtype = hadamard_dtype
        self.num_chunks = 1
        self.tmp = None

    def forward(self, q, num_kv_heads, attention_kwargs, layer_idx):
        batch_size, q_len, num_qo_heads, head_dim = q.shape
        assert q_len == 1
        q = q.view(batch_size, num_qo_heads, head_dim)
        if self.hadamard_dtype is not None:
            q = matmul_had_cuda(q, dtype=self.hadamard_dtype)
        attn_output = torch.empty_like(q)
        if self.disable_quant:
            batch_decode_f16(
                attn_output, q, **attention_kwargs, layer_idx=layer_idx
            )
        else:
            batch_decode_i4(
                attn_output, q, **attention_kwargs, layer_idx=layer_idx,
                tmp=self.tmp, num_chunks=self.num_chunks,
            )
        attn_output = attn_output.unsqueeze(1)
        return attn_output


class MultiLayerPagedKVCache4Bit(Cache):
    def __init__(
        self, batch_size, page_size, max_seq_len, 
        device, n_layers, num_heads, head_dim,
        disable_quant=False, hadamard_dtype=torch.float16, group_size=1, lac=False):
        # learned activation clipping: unused today (the clip factors below are
        # pinned to 1); it selects the unfused quantize+pack path when enabled
        self.lac = lac
        self.page_size = page_size
        self.batch_size = batch_size
        max_page_cnt = self.page_cnt_from_length(max_seq_len)
        self.disable_quant = disable_quant
        self.pages = torch.empty(
            (
                max_page_cnt * batch_size, 
                n_layers, 
                2, 
                num_heads, 
                page_size, 
                head_dim if disable_quant else head_dim // 2 
            ), 
            dtype=torch.float16 if disable_quant else torch.uint8, device=device)
        
        self.scales = torch.empty((max_page_cnt * batch_size, n_layers, 2, num_heads, page_size,  2), dtype=torch.float16, device=device)
        self.page_size = page_size
        self.max_seq_len = max_seq_len
        self._needs_init = [True] * n_layers
        self.length = 0
        self.device = device
        self.hadamard_dtype = hadamard_dtype
        self.group_size = group_size
        self._stub = _AttentionStub(
            self.page_size, device, n_layers,
            disable_quant=self.disable_quant,
            hadamard_dtype=self.hadamard_dtype)

        # Persistent index tensors for the flashinfer kernels.  They used to be
        # rebuilt (with a device->host sync) twice per layer, i.e. 64x per decode
        # token; every value is derivable on the host from `self.length`, so the
        # buffers are allocated once and refreshed in place.
        #
        # kv_indices maps (request, page) -> physical page; it depends on the
        # current page count, so it is rewritten in place when that changes
        # (once per page_size tokens) rather than reallocated.
        #
        # Every tensor in the spec dict is a fixed allocation refreshed in
        # place, so the addresses a CUDA graph captures stay valid for the whole
        # decode -- including across page-count changes.
        self._kv_indices_full = torch.empty(
            self.pages.shape[0], device=device, dtype=torch.int32)
        self._spec_page_cnt = None
        self._last_page_offset = torch.empty(batch_size, device=device, dtype=torch.int32)
        self._indptr_base = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
        self._kv_indptr_buf = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
        self._specs_frozen = False
        self._spec_cache = {
            "kv_data": self.pages,
            "kv_param": self.scales,
            "kv_indptr": self._kv_indptr_buf,
            "kv_indices": self._kv_indices_full,
            "last_page_offset": self._last_page_offset,
        }
        self._configure_decode_split(batch_size, num_heads, head_dim, max_page_cnt, device)

    def _configure_decode_split(self, batch_size, num_heads, head_dim, max_page_cnt, device):
        """Split each sequence across several blocks in the decode kernel.

        One block per (request, head) leaves batch-1 decode with num_heads
        blocks, far short of what saturates the GPU, so the sequence is cut into
        chunks that each produce a partial softmax state.  The chunk count is
        fixed for the cache's lifetime (it is baked into a CUDA graph) and
        bounded by the page count, since a chunk is a slice of the page range.
        """
        if self.disable_quant:
            return
        target_blocks = 2 * torch.cuda.get_device_properties(device).multi_processor_count
        chunks = max(1, -(-target_blocks // (batch_size * num_heads)))
        chunks = min(chunks, max(1, max_page_cnt))
        self._stub.num_chunks = chunks
        if chunks > 1:
            self._stub.tmp = torch.empty(
                (batch_size, num_heads, chunks, head_dim + 4),
                dtype=torch.float32, device=device)

    def page_cnt_from_length(self, length):
        return (length + self.page_size - 1) // self.page_size
    
    def _ensure_page_cnt_per_batch(self, expected_page_cnt_per_batch):
        expected_total = expected_page_cnt_per_batch * self.batch_size
        current_total = self.pages.shape[0]
        if expected_total <= current_total:
            return

        grow_to_total = max(expected_total, current_total * 2)
        add_total = grow_to_total - current_total

        new_pages = torch.empty(
            (add_total, *self.pages.shape[1:]),
            dtype=self.pages.dtype,
            device=self.pages.device,
        )
        self.pages = torch.cat([self.pages, new_pages], dim=0)

        new_scales = torch.empty(
            (add_total, *self.scales.shape[1:]),
            dtype=self.scales.dtype,
            device=self.scales.device,
        )
        self.scales = torch.cat([self.scales, new_scales], dim=0)

        # the page pool moved, so the captured spec dict and any CUDA graph
        # built on top of it are invalid
        self._kv_indices_full = torch.empty(
            self.pages.shape[0], device=self.device, dtype=torch.int32)
        self._spec_page_cnt = None
        self._spec_cache.update(
            kv_data=self.pages, kv_param=self.scales, kv_indices=self._kv_indices_full)

    @property
    def seen_tokens(self):
        return self.length
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        b_sz, added_length, num_heads, head_dim = key_states.shape

        orig_key_states = key_states
        orig_value_states = value_states

        if self.hadamard_dtype is not None:
            key_states = matmul_had_cuda(key_states, dtype=self.hadamard_dtype)

        if self.disable_quant:
            k_param = torch.cat(
                [key_states.new_ones((b_sz, added_length, num_heads, 1)),
                 key_states.new_zeros((b_sz, added_length, num_heads, 1))], dim=-1)
            v_param = k_param
        elif not self.lac:
            # fused amin/amax + scale/zero + quantize + pack, and the [scale,
            # zero] pair is written in the layout the flashinfer kv kernels want
            key_states, k_param = signrot_kernel.kv_quant_pack(key_states)
            value_states, v_param = signrot_kernel.kv_quant_pack(value_states)
        else:
            # kclip_factor_a_max = torch.sigmoid(kclip_factor_a_max)
            # kclip_factor_a_min = torch.sigmoid(kclip_factor_a_min)
            # vclip_factor_a_max = torch.sigmoid(vclip_factor_a_max)
            # vclip_factor_a_min = torch.sigmoid(vclip_factor_a_min)
            kclip_factor_a_max = torch.tensor(1)
            kclip_factor_a_min = torch.tensor(1)
            vclip_factor_a_max = torch.tensor(1)
            vclip_factor_a_min = torch.tensor(1)

            key_states, k_scale, k_zero = asym_quantize_and_pack_i4(
                key_states,
                clip_factor_a_max=kclip_factor_a_max,
                clip_factor_a_min=kclip_factor_a_min,
                lac=True,
            )
            value_states, v_scale, v_zero = asym_quantize_and_pack_i4(
                value_states,
                clip_factor_a_max=vclip_factor_a_max,
                clip_factor_a_min=vclip_factor_a_min,
                lac=True,
            )
            k_param = torch.cat([k_scale, k_zero], dim=-1)
            v_param = torch.cat([v_scale, v_zero], dim=-1)

        if self.group_size > 1:
            # Repeat the quantized states
            key_states = key_states.repeat_interleave(self.group_size, dim=2)
            value_states = value_states.repeat_interleave(self.group_size, dim=2)
            k_param = k_param.repeat_interleave(self.group_size, dim=-2)
            v_param = v_param.repeat_interleave(self.group_size, dim=-2)
            # Update num_heads to match after repeat
            num_heads = num_heads * self.group_size

        k_param = k_param.view(self.batch_size * added_length, num_heads, 2)
        v_param = v_param.view(self.batch_size * added_length, num_heads, 2)

        quantized_head_dim = self.pages.shape[-1]

        assert b_sz == self.batch_size
        if layer_idx == 0:
            current_length = self.length
            new_length = current_length + added_length
            self._ensure_page_cnt_per_batch(self.page_cnt_from_length(new_length))
            self.length = new_length
        attention_mask = cache_kwargs.get("attention_mask")
        # one spec build per layer instead of two (append + decode both used to
        # rebuild it from scratch)
        specs = self.get_cache_specs_for_flash_infer(attention_mask)
        if self._needs_init[layer_idx]:
            self._needs_init[layer_idx] = False
            if attention_mask is not None:
                nonzero_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten().view(-1, 1)
                key_states = key_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                value_states = value_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                key_states = torch.gather(key_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))
                value_states = torch.gather(value_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))

                k_param = k_param.view(self.batch_size * added_length, num_heads * 2)
                v_param = v_param.view(self.batch_size * added_length, num_heads * 2)
                k_param = torch.gather(k_param, 0, nonzero_indices.expand(-1, num_heads * 2))
                v_param = torch.gather(v_param, 0, nonzero_indices.expand(-1, num_heads * 2))

                seqlens_in_batch = torch.nn.functional.pad(torch.cumsum(attention_mask.sum(dim=-1, dtype=torch.int32), dim=0, dtype=torch.int32), (1, 0))
            else:
                seqlens_in_batch = torch.arange(self.batch_size + 1, device=self.device, dtype=torch.int) * added_length

            init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
            init_kv(
                **specs,
                k=key_states.view(-1, num_heads, quantized_head_dim), 
                v=value_states.view(-1, num_heads, quantized_head_dim), 
                k_param=k_param.view(-1, num_heads, 2), 
                v_param=v_param.view(-1, num_heads, 2),
                seqlen_indptr=seqlens_in_batch,
                layer_idx=layer_idx
            )
            return orig_key_states, orig_value_states
        else:
            assert added_length == 1
            append_kv = append_kv_f16 if self.disable_quant else append_kv_i4
            append_kv(
                **specs,
                k=key_states.view(self.batch_size, num_heads, quantized_head_dim), 
                v=value_states.view(self.batch_size, num_heads, quantized_head_dim), 
                k_param=k_param.view(-1, num_heads, 2), 
                v_param=v_param.view(-1, num_heads, 2),
                layer_idx=layer_idx,
            )
        return functools.partial(
            self._stub.forward,
            num_kv_heads=num_heads,
            attention_kwargs=specs,
            layer_idx=layer_idx,
        )
    
    def _set_page_cnt(self, page_cnt: int):
        """Point the index buffers at a `page_cnt`-pages-per-request layout.

        Physical page for (request b, page j) is j * batch_size + b, matching how
        `_ensure_page_cnt_per_batch` grows the pool.  Only rewritten when the
        page count actually changes.
        """
        torch.mul(self._indptr_base, page_cnt, out=self._kv_indptr_buf)
        if self._spec_page_cnt == page_cnt:
            return
        self._spec_page_cnt = page_cnt
        idx = (
            torch.arange(page_cnt, device=self.device, dtype=torch.int32) * self.batch_size
        ).unsqueeze(0) + torch.arange(
            self.batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        self._kv_indices_full[: page_cnt * self.batch_size].copy_(idx.reshape(-1))

    def refresh_specs(self, length: int | None = None):
        """Refresh the flashinfer index tensors for a uniform-length batch.

        Everything here is derived from the host-side `self.length`, so this
        costs two tiny in-place writes and no device->host sync.  Kept callable
        on its own so a CUDA-graph runner can refresh the buffers between
        replays -- the graph itself must not contain these writes, since their
        values change every step.
        """
        if length is None:
            length = self.length
        page_cnt = self.page_cnt_from_length(length)
        page_ptr = length % self.page_size
        if length != 0 and page_ptr == 0:
            page_ptr = self.page_size
        self._last_page_offset.fill_(page_ptr)
        self._set_page_cnt(page_cnt)
        return self._spec_cache

    def freeze_specs(self, frozen=True):
        """Stop refreshing the index buffers inside the model forward.

        Used while capturing/replaying a CUDA graph: the caller refreshes them
        once per step from outside the graph instead.
        """
        self._specs_frozen = frozen

    def get_cache_specs_for_flash_infer(self, attention_mask):
        if attention_mask is None:
            return self._spec_cache if self._specs_frozen else self.refresh_specs()

        # Padded prefill: the per-sequence lengths only exist on device.  This
        # path runs once per layer per prefill, so the sync is not on the hot
        # decode path.
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        page_cnt = self.page_cnt_from_length(seqlens_in_batch)
        if (page_cnt[0] != page_cnt).any():
            raise NotImplementedError("Current implementation does not support the case where batches have different number of pages")
        page_cnt = int(page_cnt[0])
        page_ptr = seqlens_in_batch % self.page_size
        page_ptr = torch.where((seqlens_in_batch != 0) & (page_ptr == 0), self.page_size, page_ptr)
        self._last_page_offset.copy_(page_ptr)
        self._set_page_cnt(page_cnt)
        return self._spec_cache

    def reset(self):
        """Rewind to an empty cache without reallocating.

        The page pool and index buffers keep their addresses, so a CUDA graph
        captured against this cache stays valid across the reset.
        """
        self.length = 0
        self._needs_init = [True] * len(self._needs_init)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def to_legacy_cache(self):
        return self


class HFCacheAdapter(Cache):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_usable_length(self, kv_seq_len: int, layer_idx: int) -> int:
        return self.inner.get_usable_length(kv_seq_len, layer_idx)

    def get_seq_length(self) -> int:
        return self.inner.get_seq_length()

    def get_max_length(self) -> int:
        return self.inner.get_max_length()

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        return self.inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def to_legacy_cache(self):
        if hasattr(self.inner, "to_legacy_cache"):
            return self.inner.to_legacy_cache()
        return super().to_legacy_cache()
