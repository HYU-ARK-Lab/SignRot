"""MoE block over stacked int4 expert weights.

Replaces the HF Qwen3MoeSparseMoeBlock per-expert Python loop, whose
`expert_hit.nonzero()` forces a device->host sync per layer and whose top-8
M=1 GEMM chains dominate batch-1 decode.  All routing here stays on the GPU
with fixed-shape ops (argsort/index_add_/cumsum), so the forward is CUDA-graph
capturable and one kernel serves every expert:

  router -> permute rows by expert -> quantize each token once (its top-k
  experts and gate/up all read the same codes) -> grouped gate/up GEMM ->
  swiglu -> Hadamard over all rows at once (the transform is
  expert-independent; only the had_adj sign vector differs per expert, fused
  into the down-proj quantizer as a [E, inter] table indexed by each row's
  expert) -> grouped down GEMM -> weighted un-permute.

The GEMM/dequant stage has two implementations over the same permuted
buffers, bit-identical by construction: the grouped kernels (default), and a
host-synced per-expert loop over the dense kernels (`use_grouped=False`, the
equivalence oracle and debug fallback).
"""

import torch

import signrot.functional.kernel as kernel
from signrot.functional.hadamard import get_hadK, matmul_hadU_cuda


class SignRotMoEBlock(torch.nn.Module):
    def __init__(self, gate, *, top_k, norm_topk_prob,
                 w_gate, s_gate, w_up, s_up, w_down, s_down,
                 sign_down=None, wsum_gate=None, wsum_up=None, wsum_down=None,
                 asym=False, clip_ratio=1.0, fp32_had=False, use_grouped=True):
        super().__init__()
        self.gate = gate  # router linear (16-bit activations), left untouched
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.asym = asym
        self.clip_ratio = clip_ratio
        self.fp32_had = fp32_had
        self.use_grouped = use_grouped

        self.num_experts, self.intermediate_size = w_gate.shape[0], w_gate.shape[1]
        self.hidden_size = w_gate.shape[2] * 2
        assert w_down.shape == (self.num_experts, self.hidden_size, self.intermediate_size // 2)

        self.register_buffer("w_gate", w_gate, persistent=False)
        self.register_buffer("w_up", w_up, persistent=False)
        self.register_buffer("w_down", w_down, persistent=False)
        self.register_buffer("s_gate", s_gate, persistent=False)
        self.register_buffer("s_up", s_up, persistent=False)
        self.register_buffer("s_down", s_down, persistent=False)
        self.register_buffer("sign_down", sign_down, persistent=False)
        # int4 row sums for the asym zero correction; may be left None and are
        # then computed lazily on the first forward (sum_i4 is a CUDA kernel,
        # but conversion can happen while the model is still on the CPU)
        self.register_buffer("wsum_gate", wsum_gate, persistent=False)
        self.register_buffer("wsum_up", wsum_up, persistent=False)
        self.register_buffer("wsum_down", wsum_down, persistent=False)

        had_K, self.K = get_hadK(self.intermediate_size)
        if had_K is not None:
            # keep it on the weights' device: matmul_hadU_cuda would otherwise
            # re-upload the CPU tensor every forward, which both wastes a copy
            # and (pageable H2D) invalidates CUDA-graph capture
            had_K = had_K.to(w_gate.device, torch.float32 if fp32_had else torch.float16)
        self.register_buffer("had_K", had_K, persistent=False)

        self._grouped_dev = None

    @staticmethod
    def stack_experts(linears, need_wsum):
        """Stack per-expert Linear4bit weights/scales (and int4 row sums)."""
        w = torch.stack([m.weight for m in linears])
        s = torch.stack([m.weight_scales.view(-1).to(torch.float16) for m in linears])
        wsum = None
        if need_wsum:
            wsum = torch.stack([kernel.sum_i4(m.weight).view(-1) for m in linears])
        return w.contiguous(), s.contiguous(), wsum

    def _ensure_wsum(self):
        if not self.asym or self.wsum_gate is not None:
            return
        e = self.num_experts
        self.wsum_gate = kernel.sum_i4(self.w_gate).view(e, -1)
        self.wsum_up = kernel.sum_i4(self.w_up).view(e, -1)
        self.wsum_down = kernel.sum_i4(self.w_down).view(e, -1)

    def _quantize(self, x):
        if self.asym:
            q, s, z = kernel.asym_quant_dynamic(x, self.clip_ratio)
            return q, s, z.to(torch.int8)
        q, s = kernel.sym_quant_dynamic(x, self.clip_ratio)
        return q, s, None

    # ------------------------------------------------------------- grouped path

    def _grouped_consts(self, device):
        """Per-device constant tensors for the grouped GEMM calls: stacked-
        weight base pointers, leading dims and (m, n, k) problem-size templates
        whose m column is refreshed from `counts` each forward."""
        if self._grouped_dev == device:
            return self._gc
        e = self.num_experts
        h, i = self.hidden_size, self.intermediate_size
        arange = torch.arange(e, dtype=torch.int64, device=device)

        def consts(w, n, k):
            ps = torch.empty((e, 3), dtype=torch.int32, device=device)
            ps[:, 1] = n
            ps[:, 2] = k
            return {
                "ps": ps,
                "ptr_b": w.data_ptr() + arange * w.stride(0),
                "ld_ab": torch.full((e,), k, dtype=torch.int64, device=device),
                "ld_c": torch.full((e,), n, dtype=torch.int64, device=device),
            }

        self._gc = {
            "gate": consts(self.w_gate, i, h),
            "up": consts(self.w_up, i, h),
            "down": consts(self.w_down, h, i),
        }
        self._grouped_dev = device
        return self._gc

    def _gemm_dequant_grouped(self, q, s, z, eid, row_off, w, s_w, wsum, gc):
        m, n = q.shape[0], w.shape[1]
        c = torch.empty((m, n), dtype=torch.int32, device=q.device)
        ptr_a = q.data_ptr() + row_off * q.shape[1]  # packed rows: K/2 bytes each
        ptr_c = c.data_ptr() + row_off * (n * 4)
        kernel.matmul_grouped(gc["ps"], ptr_a, gc["ptr_b"], ptr_c,
                              gc["ld_ab"], gc["ld_ab"], gc["ld_c"],
                              signed_a=z is None)
        return kernel.sym_dequant_moe(c, s, s_w, eid, z, wsum)

    # ---------------------------------------------------- reference (loop) path

    def _gemm_dequant_loop(self, q, s, z, counts, w, s_w, wsum):
        """Per-expert loop over the dense single-GEMM kernels.

        Host-syncs on counts; the equivalence oracle for the grouped path
        (identical inputs, bit-identical GEMM/dequant results).
        """
        out = torch.empty(q.shape[0], w.shape[1], dtype=torch.float16, device=q.device)
        row = 0
        for e, m in enumerate(counts.tolist()):
            if m == 0:
                continue
            rows = slice(row, row + m)
            row += m
            if z is not None:
                i32 = kernel.matmul_ui4_si4(q[rows], w[e])
                i32 -= kernel.matmul_colrow_i8_i32(z[rows], wsum[e].view(1, -1))
            else:
                i32 = kernel.matmul_si4_si4(q[rows], w[e])
            out[rows] = kernel.sym_dequant(i32, s[rows], s_w[e].view(-1, 1))
        return out

    # ---------------------------------------------------------------- forward

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        x_dtype = x.dtype
        self._ensure_wsum()

        router_logits = self.gate(x)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x_dtype)

        # sort the (token, choice) rows by expert; every op here is fixed-shape
        # and sync-free (index_add_ instead of bincount, which reads max() back
        # to the host)
        flat_e = selected.reshape(-1)
        perm = torch.argsort(flat_e, stable=True)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=x.device)
        counts.index_add_(0, flat_e, torch.ones_like(flat_e, dtype=torch.int32))
        tok = perm // self.top_k
        eid = flat_e[perm].to(torch.int32)

        # each token's top-k experts and its gate/up projections all consume the
        # same activation: quantize once per token, gather the packed codes
        q_tok, s_tok, z_tok = self._quantize(x.to(torch.float16))
        q_a, s_a = q_tok[tok], s_tok[tok]
        z_a = z_tok[tok] if z_tok is not None else None

        if self.use_grouped:
            gc = self._grouped_consts(x.device)
            row_off = torch.zeros(self.num_experts, dtype=torch.int64, device=x.device)
            row_off[1:] = torch.cumsum(counts[:-1], 0)
            gc["gate"]["ps"][:, 0] = counts
            gc["up"]["ps"][:, 0] = counts
            gc["down"]["ps"][:, 0] = counts
            gate_h = self._gemm_dequant_grouped(
                q_a, s_a, z_a, eid, row_off, self.w_gate, self.s_gate, self.wsum_gate, gc["gate"])
            up_h = self._gemm_dequant_grouped(
                q_a, s_a, z_a, eid, row_off, self.w_up, self.s_up, self.wsum_up, gc["up"])
        else:
            gate_h = self._gemm_dequant_loop(
                q_a, s_a, z_a, counts, self.w_gate, self.s_gate, self.wsum_gate)
            up_h = self._gemm_dequant_loop(
                q_a, s_a, z_a, counts, self.w_up, self.s_up, self.wsum_up)
        inter = kernel.swiglu(gate_h, up_h)

        # the Hadamard transform is identical for every expert; only the
        # had_adj sign differs, indexed per row from the [E, inter] table
        h = inter.float() if self.fp32_had else inter
        h = matmul_hadU_cuda(h, self.had_K, self.K).to(torch.float16)

        if self.use_grouped:
            if self.asym:
                q_d, s_d, z_d = kernel.asym_quant_dynamic_moe(
                    h, self.sign_down, eid, self.clip_ratio)
                z_d = z_d.to(torch.int8)
            else:
                q_d, s_d = kernel.sym_quant_dynamic_moe(
                    h, self.sign_down, eid, self.clip_ratio)
                z_d = None
            y = self._gemm_dequant_grouped(
                q_d, s_d, z_d, eid, row_off, self.w_down, self.s_down, self.wsum_down, gc["down"])
        else:
            if self.sign_down is not None:
                h = h * self.sign_down[eid]
            q_d, s_d, z_d = self._quantize(h)
            y = self._gemm_dequant_loop(
                q_d, s_d, z_d, counts, self.w_down, self.s_down, self.wsum_down)

        # deterministic un-permute + weighted sum (no index_add_/atomics, so a
        # captured graph replays bit-identically)
        inv = torch.argsort(perm)
        y = y.to(x_dtype)[inv].view(-1, self.top_k, hidden_dim)
        out = (y * routing_weights.unsqueeze(-1)).sum(dim=1)
        return out.view(batch_size, seq_len, hidden_dim), router_logits
