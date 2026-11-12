#include <stream.h>
#include <gemm_grouped.h>

#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <gemm_grouped_mixed.h>

namespace {

// Same operand types, epilogue and instruction as the dense int4 Gemm in
// gemm.cu, so per-element results are bit-identical to a per-expert loop.
// The threadblock tile is 128x128 (vs the dense default's 128x256): expert
// problems are as small as M=1, and a narrower N tile wastes fewer lanes.
// DefaultGemmGrouped only supplies the type plumbing (Mma/Epilogue); the
// kernel itself is the vendored GemmGroupedMixedInput, because the upstream
// kernel body does not compile when ElementA != ElementB (see
// gemm_grouped_mixed.h).  Its operator() is never instantiated here.
template<typename ElementA>
using DefaultKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        32,
        cutlass::int4b_t,
        cutlass::layout::ColumnMajor,
        cutlass::ComplexTransform::kNone,
        32,
        int32_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::epilogue::thread::LinearCombinationClamp<
                int32_t, 128 / cutlass::sizeof_bits<int32_t>::value, int32_t, float>,
        // unused by grouped kernels; passed to match the kernel APIs
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

template<typename ElementA>
using GroupedKernel = cutlass::gemm::kernel::GemmGroupedMixedInput<
        typename DefaultKernel<ElementA>::Mma,
        typename DefaultKernel<ElementA>::Epilogue,
        typename DefaultKernel<ElementA>::ThreadblockSwizzle,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
        DefaultKernel<ElementA>::kTransposed>;

template<typename ElementA>
void run_grouped(
        const cutlass::gemm::GemmCoord *problem_sizes,
        int problem_count,
        const int64_t *ptr_A,
        const int64_t *ptr_B,
        const int64_t *ptr_C,
        int64_t *lda,
        int64_t *ldb,
        int64_t *ldc
)
{
    using Gemm = cutlass::gemm::device::GemmGrouped<GroupedKernel<ElementA>>;

    // occupancy-based persistent grid size; depends only on the GPU model
    static int threadblock_count = Gemm::sufficient();
    ensure(threadblock_count > 0, "gemm_grouped: zero available threadblocks");

    typename Gemm::Arguments arguments(
            const_cast<cutlass::gemm::GemmCoord *>(problem_sizes),
            problem_count,
            threadblock_count,
            {1, 0},
            reinterpret_cast<ElementA **>(const_cast<int64_t *>(ptr_A)),
            reinterpret_cast<cutlass::int4b_t **>(const_cast<int64_t *>(ptr_B)),
            // beta == 0: the C operand is never read, reuse the output pointers
            reinterpret_cast<int32_t **>(const_cast<int64_t *>(ptr_C)),
            reinterpret_cast<int32_t **>(const_cast<int64_t *>(ptr_C)),
            lda, ldb, ldc, ldc,
            /*host_problem_sizes=*/nullptr  // not needed with kDeviceOnly
    );

    Gemm gemmOp;
    auto status = gemmOp.initialize(arguments, nullptr, SIGNROT_STREAM);
    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
    status = gemmOp.run(SIGNROT_STREAM);
    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}

}  // namespace

void matmul_grouped_si4_si4_host(
        const cutlass::gemm::GemmCoord *problem_sizes,
        int problem_count,
        const int64_t *ptr_A,
        const int64_t *ptr_B,
        const int64_t *ptr_C,
        int64_t *lda,
        int64_t *ldb,
        int64_t *ldc
)
{
    run_grouped<cutlass::int4b_t>(problem_sizes, problem_count, ptr_A, ptr_B, ptr_C, lda, ldb, ldc);
}

void matmul_grouped_ui4_si4_host(
        const cutlass::gemm::GemmCoord *problem_sizes,
        int problem_count,
        const int64_t *ptr_A,
        const int64_t *ptr_B,
        const int64_t *ptr_C,
        int64_t *lda,
        int64_t *ldb,
        int64_t *ldc
)
{
    run_grouped<cutlass::uint4b_t>(problem_sizes, problem_count, ptr_A, ptr_B, ptr_C, lda, ldb, ldc);
}
