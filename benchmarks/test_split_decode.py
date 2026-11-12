"""The split-KV decode kernel must produce the same attention output as the
single-block kernel it replaces.

Splitting changes the order in which softmax states are combined, so the result
is not bit-identical; it is checked against the unsplit kernel and against an
fp32 reference computed from the same dequantized cache.

Run: .venv/bin/python benchmarks/test_split_decode.py
"""

import math

import torch

from signrot.transformers import MultiLayerPagedKVCache4Bit
from signrot.transformers.kv_cache import batch_decode_i4, unpack_i4_and_asym_dequantize

DEV = "cuda"
FAILED = []


def reference_attention(cache, q, layer_idx, seq_len):
    """fp32 attention over the dequantized paged cache."""
    b, h, d = q.shape
    page_size = cache.page_size
    pages = cache.pages
    scales = cache.scales
    outs = []
    for bi in range(b):
        ks, vs = [], []
        for t in range(seq_len):
            page = (t // page_size) * cache.batch_size + bi
            slot = t % page_size
            kq = pages[page, layer_idx, 0, :, slot, :]
            vq = pages[page, layer_idx, 1, :, slot, :]
            kp = scales[page, layer_idx, 0, :, slot, :]
            vp = scales[page, layer_idx, 1, :, slot, :]
            ks.append(unpack_i4_and_asym_dequantize(kq, kp[:, 0:1], kp[:, 1:2]))
            vs.append(unpack_i4_and_asym_dequantize(vq, vp[:, 0:1], vp[:, 1:2]))
        k = torch.stack(ks, 0).float()   # [T, H, D]
        v = torch.stack(vs, 0).float()
        qi = q[bi].float()               # [H, D]
        logits = torch.einsum("hd,thd->ht", qi, k) / math.sqrt(d)
        w = torch.softmax(logits, dim=-1)
        outs.append(torch.einsum("ht,thd->hd", w, v))
    return torch.stack(outs, 0)


def check(name, got, ref, atol):
    diff = (got.float() - ref.float()).abs().max().item()
    ok = diff <= atol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:44s} max abs diff {diff:.3e} (tol {atol:.1e})")
    if not ok:
        FAILED.append(name)


def run_case(batch, seq_len, page_size, n_heads=32, head_dim=128, n_layers=2, layer_idx=1):
    torch.manual_seed(0)
    max_len = seq_len
    cache = MultiLayerPagedKVCache4Bit(
        batch_size=batch, page_size=page_size, max_seq_len=max_len, device=DEV,
        n_layers=n_layers, num_heads=n_heads, head_dim=head_dim,
        disable_quant=False, hadamard_dtype=None, group_size=1)

    k = torch.randn(batch, seq_len, n_heads, head_dim, device=DEV, dtype=torch.float16)
    v = torch.randn(batch, seq_len, n_heads, head_dim, device=DEV, dtype=torch.float16)
    for li in range(n_layers):
        cache.update(k, v, li, {"attention_mask": None})

    q = torch.randn(batch, n_heads, head_dim, device=DEV, dtype=torch.float16)
    specs = cache.refresh_specs()

    unsplit = torch.empty_like(q)
    batch_decode_i4(unsplit, q, **specs, layer_idx=layer_idx)

    chunks = cache._stub.num_chunks
    ref = reference_attention(cache, q, layer_idx, seq_len)
    check(f"b={batch} T={seq_len} P={page_size} unsplit vs fp32 ref",
          unsplit, ref, 3e-2)

    for nc in sorted({2, 4, 8, chunks, cache.page_cnt_from_length(seq_len)}):
        if nc < 2 or nc > cache.page_cnt_from_length(seq_len):
            continue
        tmp = torch.empty((batch, n_heads, nc, head_dim + 4),
                          dtype=torch.float32, device=DEV)
        split = torch.empty_like(q)
        batch_decode_i4(split, q, **specs, layer_idx=layer_idx,
                        tmp=tmp, num_chunks=nc)
        check(f"b={batch} T={seq_len} P={page_size} chunks={nc} vs unsplit",
              split, unsplit, 2e-3)
        check(f"b={batch} T={seq_len} P={page_size} chunks={nc} vs fp32 ref",
              split, ref, 3e-2)

    del cache
    torch.cuda.empty_cache()


print("=== split-KV decode vs single-block decode ===")
run_case(batch=1, seq_len=1024, page_size=256)
run_case(batch=1, seq_len=1000, page_size=256)   # ragged last page
run_case(batch=4, seq_len=513, page_size=128)    # chunks > pages for some
run_case(batch=2, seq_len=256, page_size=256)    # single page: chunking is a no-op
run_case(batch=1, seq_len=2048, page_size=256)

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} checks")
    for f in FAILED:
        print("  -", f)
    raise SystemExit(1)
print("split decode matches the unsplit kernel and the fp32 reference")
