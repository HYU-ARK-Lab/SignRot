#pragma once
#include <common.h>
#include <cutlass/gemm_coord.h>

// Grouped int4 GEMM over stacked MoE expert weights: C_e = A_e @ B_e^T for
// every expert e in one persistent-grid launch.  All per-problem metadata
// (sizes, pointers, leading dims) lives in device memory and is scheduled on
// the device (GroupScheduleMode::kDeviceOnly), so the launch shape does not
// depend on the routing -- required for CUDA-graph capture of a decode step.
// Pointer arrays are passed as int64 device addresses and reinterpreted.

void matmul_grouped_si4_si4_host(
        const cutlass::gemm::GemmCoord *problem_sizes,  // device, [problem_count]
        int problem_count,
        const int64_t *ptr_A,  // device, addresses of packed int4 A rows
        const int64_t *ptr_B,  // device, addresses of packed int4 B slabs
        const int64_t *ptr_C,  // device, addresses of int32 C rows
        int64_t *lda,
        int64_t *ldb,
        int64_t *ldc
);

void matmul_grouped_ui4_si4_host(
        const cutlass::gemm::GemmCoord *problem_sizes,
        int problem_count,
        const int64_t *ptr_A,
        const int64_t *ptr_B,
        const int64_t *ptr_C,
        int64_t *lda,
        int64_t *ldb,
        int64_t *ldc
);
