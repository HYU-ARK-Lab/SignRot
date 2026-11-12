"""Expert-indexed fused MoE kernels vs the eager composition they replace.

sym/asym_quant_dynamic_moe must reproduce `x * sign[eid]` followed by
sym/asym_quant_dynamic bit-exactly (codes, scales, zeros), and sym_dequant_moe
must reproduce matmul_colrow_i8_i32 + int32 subtract + sym_dequant bit-exactly.

Run: .venv/bin/python benchmarks/test_moe_kernels.py
"""

import sys

import torch

import signrot.functional.kernel as kernel

DEV = "cuda"

failures = []


def report(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<52} {detail}")
    if not ok:
        failures.append(name)


def make_rows(rows, cols, experts):
    x = torch.randn(rows, cols, device=DEV, dtype=torch.float16)
    eid = torch.randint(0, experts, (rows,), dtype=torch.int32, device=DEV)
    sign = torch.randint(0, 2, (experts, cols), dtype=torch.int8, device=DEV) * 2 - 1
    sign[-1] = 1  # an identity (never-calibrated) expert
    return x, eid, sign


@torch.no_grad()
def check_quant(name, rows, cols, experts, asym, clip_ratio, with_sign):
    x, eid, sign = make_rows(rows, cols, experts)
    sign_arg = sign if with_sign else None
    x_ref = x * sign[eid.long()] if with_sign else x

    if asym:
        q, s, z = kernel.asym_quant_dynamic_moe(x, sign_arg, eid, clip_ratio)
        q_ref, s_ref, z_ref = kernel.asym_quant_dynamic(x_ref, clip_ratio)
        ok = torch.equal(q, q_ref) and torch.equal(s, s_ref) and torch.equal(z, z_ref)
    else:
        q, s = kernel.sym_quant_dynamic_moe(x, sign_arg, eid, clip_ratio)
        q_ref, s_ref = kernel.sym_quant_dynamic(x_ref, clip_ratio)
        ok = torch.equal(q, q_ref) and torch.equal(s, s_ref)
    report(name, ok, "bit-exact" if ok else "MISMATCH")


@torch.no_grad()
def check_dequant(name, rows, cols, experts, asym):
    q = torch.randint(-(2**20), 2**20, (rows, cols), dtype=torch.int32, device=DEV)
    s_row = (torch.rand(rows, 1, device=DEV, dtype=torch.float16) + 0.1) / 64
    w_scales = (torch.rand(experts, cols, device=DEV, dtype=torch.float16) + 0.1) / 64
    eid = torch.randint(0, experts, (rows,), dtype=torch.int32, device=DEV)

    zero_row = w_sum = None
    q_adj = q
    if asym:
        zero_row = torch.randint(0, 16, (rows,), dtype=torch.int8, device=DEV)
        w_sum = torch.randint(-8 * cols, 7 * cols, (experts, cols),
                              dtype=torch.int32, device=DEV)
        q_adj = q.clone()
        for r in range(rows):  # reference: rank-1 int32 correction per row
            q_adj[r] -= zero_row[r].int() * w_sum[eid[r].long()]

    out = kernel.sym_dequant_moe(q, s_row, w_scales, eid, zero_row, w_sum)
    ref = torch.empty_like(out)
    for r in range(rows):
        ref[r] = kernel.sym_dequant(q_adj[r:r + 1], s_row[r:r + 1],
                                    w_scales[eid[r].long()].view(-1, 1))
    ok = torch.equal(out, ref)
    report(name, ok, "bit-exact" if ok else "MISMATCH")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    print("=== fused sign + dynamic quant (vs x * sign[eid] -> quant) ===")
    for asym in (False, True):
        tag = "asym" if asym else "sym"
        check_quant(f"{tag} 8x2048 sign", 8, 2048, 128, asym, 1.0, True)
        check_quant(f"{tag} 8x768 sign (down_proj decode)", 8, 768, 128, asym, 1.0, True)
        check_quant(f"{tag} 4096x768 sign (prefill)", 4096, 768, 128, asym, 1.0, True)
        check_quant(f"{tag} 33x768 no-sign", 33, 768, 128, asym, 1.0, False)
        check_quant(f"{tag} 128x768 sign clip 0.9", 128, 768, 128, asym, 0.9, True)
        check_quant(f"{tag} 5x192 sign (small geometry)", 5, 192, 16, asym, 1.0, True)

    print("=== expert-indexed dequant (+ fused zero correction) ===")
    for asym in (False, True):
        tag = "asym" if asym else "sym"
        check_dequant(f"{tag} 8x768", 8, 768, 128, asym)
        check_dequant(f"{tag} 8x2048", 8, 2048, 128, asym)
        check_dequant(f"{tag} 200x2048", 200, 2048, 128, asym)

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("\nall MoE kernel checks passed")


if __name__ == "__main__":
    main()
