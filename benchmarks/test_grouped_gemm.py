"""matmul_grouped vs a per-problem loop of the dense int4 GEMMs.

The grouped kernel uses the same operand types, instruction and epilogue as
matmul_si4_si4 / matmul_ui4_si4, so every problem's int32 output must be
bit-exact.  Exercises zero-row problems (never-routed experts), heavy skew,
M=1 rows (decode) and both MoE projection geometries.

Run: .venv/bin/python benchmarks/test_grouped_gemm.py
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


@torch.no_grad()
def check(name, counts, n, k, signed_a):
    e = len(counts)
    m_total = sum(counts)
    a = torch.randint(0, 256, (m_total, k // 2), dtype=torch.uint8, device=DEV)
    b = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=DEV)
    c = torch.empty((m_total, n), dtype=torch.int32, device=DEV)

    counts_t = torch.tensor(counts, dtype=torch.int32, device=DEV)
    offsets = torch.zeros(e, dtype=torch.int64, device=DEV)
    if e > 1:
        offsets[1:] = torch.cumsum(counts_t[:-1], 0)

    problem_sizes = torch.empty((e, 3), dtype=torch.int32, device=DEV)
    problem_sizes[:, 0] = counts_t
    problem_sizes[:, 1] = n
    problem_sizes[:, 2] = k

    ptr_a = a.data_ptr() + offsets * (k // 2)
    ptr_b = b.data_ptr() + torch.arange(e, dtype=torch.int64, device=DEV) * (n * k // 2)
    ptr_c = c.data_ptr() + offsets * (n * 4)
    lda = torch.full((e,), k, dtype=torch.int64, device=DEV)
    ldb = torch.full((e,), k, dtype=torch.int64, device=DEV)
    ldc = torch.full((e,), n, dtype=torch.int64, device=DEV)

    kernel.matmul_grouped(problem_sizes, ptr_a, ptr_b, ptr_c, lda, ldb, ldc, signed_a)

    dense = kernel.matmul_si4_si4 if signed_a else kernel.matmul_ui4_si4
    row = 0
    exact = True
    for i, m in enumerate(counts):
        if m == 0:
            continue
        ref = dense(a[row:row + m], b[i])
        if not torch.equal(c[row:row + m], ref):
            exact = False
            break
        row += m
    report(name, exact, "bit-exact" if exact else "MISMATCH")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    for signed_a in (True, False):
        tag = "si4" if signed_a else "ui4"
        print(f"=== A={tag} x B=si4 -> i32 ===")
        # decode: top-8 of 128 experts, M=1 each, rest zero
        check(f"{tag} decode 8x(M=1)/128 gateup", [1] * 8 + [0] * 120, 768, 2048, signed_a)
        check(f"{tag} decode 8x(M=1)/128 down", [1] * 8 + [0] * 120, 2048, 768, signed_a)
        # prefill-ish: random counts with zeros
        counts = torch.randint(0, 40, (128,)).tolist()
        counts[::7] = [0] * len(counts[::7])
        check(f"{tag} prefill random/128 gateup", counts, 768, 2048, signed_a)
        check(f"{tag} prefill random/128 down", counts, 2048, 768, signed_a)
        # heavy skew: one expert takes nearly everything
        check(f"{tag} skew [4096,3,0,1]", [4096, 3, 0, 1], 768, 2048, signed_a)
        # single problem
        check(f"{tag} single problem M=513", [513], 2048, 768, signed_a)

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("\nall grouped GEMM checks passed")


if __name__ == "__main__":
    main()
