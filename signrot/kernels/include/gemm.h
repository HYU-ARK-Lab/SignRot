// Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
#pragma once

#include <common.h>


void matmul_host(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

void matmul_host_ui4_si4(
        const UInt4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);
