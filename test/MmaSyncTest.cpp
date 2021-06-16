#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Performance.h"
#include "Utils.h"
#include "WMMA.h"
#include "Common.hpp"

#ifdef WMMA_VALIDATE_TESTS
#include <gtest/gtest.h>
#include "Reference.h" // Vanilla CPU kernel
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
#include "rocBLASReference.h" // rocBLAS GPU kernel
#endif // WMMA_VALIDATE_WITH_ROCBLAS
#endif // WMMA_VALIDATE_TESTS

static bool headerPrinted = false;

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
__global__ void __launch_bounds__(256, 1) test_mma_sync_d(uint32_t       m,
                                                          uint32_t       n,
                                                          uint32_t       k,
                                                          InputT const*  a,
                                                          InputT const*  b,
                                                          OutputT const* c,
                                                          OutputT*       d,
                                                          ComputeT       alpha,
                                                          ComputeT       beta)
{
    using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
    using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;

    int lda = std::is_same<LayoutA, row_major>::value ? k : m;
    int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
    int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
    int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

    // Create frags
    auto fragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>();
    auto fragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>();
    auto fragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, OutputT>();
    auto fragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>();

    wmma::fill_fragment(fragAcc, static_cast<ComputeT>(0));

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / AMDGCN_WAVE_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Loop over k
    for(int i = 0; i < k; i += BlockK)
    {
        int aRow = warpM * BlockM;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * BlockN;

        // Bounds checking
        if(aRow < m && aCol < k && bRow < k && bCol < n)
        {
            // Load the inputs
            wmma::load_matrix_sync(
                fragA, MappingA::dataCoord(a, lda, std::make_pair(aRow, aCol)), lda);
            wmma::load_matrix_sync(
                fragB, MappingB::dataCoord(b, ldb, std::make_pair(bRow, bCol)), ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * BlockM;
    int cCol = warpN * BlockN;

    if(cRow < m && cCol < n)
    {
        OutputT const* cOffset = c
                                 + (std::is_same<LayoutC, row_major>::value ? (cRow * ldc + cCol)
                                                                            : (cRow + cCol * ldc));
        wmma::load_matrix_sync(fragC,
                               cOffset,
                               ldc,
                               std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                       : wmma::mem_col_major);

        using CvtCIn    = amdgcn_convert<OutputT, ComputeT>;
        using UnpackC   = Unpack<OutputT, fragC.registerCount()>;
        using UnpackAcc = Unpack<ComputeT, fragAcc.registerCount()>;

        // Get ready to multiply and accum with C
        // Must convert to ComputeT and unpack
        // TODO: Add packed ops
        auto cCompute   = CvtCIn::exec(UnpackC::exec(*fragC));
        auto accCompute = UnpackAcc::exec(*fragAcc);
        static_assert(decltype(cCompute)::size() == decltype(accCompute)::size(),
                      "C and accumulator must have same register count");

#pragma unroll
        for(int i = 0; i < decltype(cCompute)::size(); ++i)
        {
            cCompute[i] = alpha * accCompute[i] + beta * cCompute[i];
        }

        // Re-configure output
        using CvtCOut = amdgcn_convert<ComputeT, OutputT>;
        using PackC   = Pack<OutputT, cCompute.size()>;
        *fragC        = PackC::exec(CvtCOut::exec(cCompute));

        OutputT* dOffset = d
                           + (std::is_same<LayoutD, row_major>::value ? (cRow * ldd + cCol)
                                                                      : (cRow + cCol * ldd));
        // Store the output
        wmma::store_matrix_sync(dOffset,
                                fragC,
                                ldd,
                                std::is_same<LayoutD, row_major>::value ? wmma::mem_row_major
                                                                        : wmma::mem_col_major);
    }
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void test_mma_sync_h(uint32_t TBlockX,
                              uint32_t TBlockY,
                              uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              ComputeT alpha,
                              ComputeT beta)
{
    if ( m < BlockM * TBlockX / AMDGCN_WAVE_SIZE || n < BlockN * TBlockY || k < BlockK )
        return;

    int lda = std::is_same<LayoutA, row_major>::value ? k : m;
    int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
    int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
    int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

    if (!headerPrinted)
    {
        std::cout << "TBlkX, TBlkY, BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, "
                    "ldd, LytA_LytB_LytC_LytD, Ti_To_Tc, elapsedMs, GFlops, GFlops/s, Efficiency(%)\n";
        headerPrinted = true;
    }
    std::cout << TBlockX << ", " << TBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
              << ", " << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
              << ", " << beta << ", " << ldc << ", " << ldd << ", "
              << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutD, row_major>::value ? "R" : "C") << ", "
              << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
              << dataTypeToString<ComputeT>();

    if ( quirks::hipcc_bug_half_packing<OutputT, LayoutC>::value )
    {
        std::cout << ", " << "na" << ", " << "na"
            << ", " << "na" << ", " << "na" << ", " << " SKIPPED" << std::endl;
        return;
    }

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n, OutputT(0));
    std::vector<OutputT> matrixD(m * n);

    MatrixUtil<LayoutA>::fill(matrixA, m, k);
    MatrixUtil<LayoutB>::fill(matrixB, k, n);
    MatrixUtil<LayoutC>::fill(matrixC, m, n);
    MatrixUtil<LayoutD>::fill(matrixD, m, n, std::numeric_limits<OutputT>::signaling_NaN());

    // Allocate and copy device memory
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_c;
    OutputT* d_d;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesC = matrixC.size() * sizeof(OutputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto gridDim
        = dim3(ceilDiv(m, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(n, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL((test_mma_sync_d<BlockM,
                                           BlockN,
                                           BlockK,
                                           InputT,
                                           OutputT,
                                           ComputeT,
                                           LayoutA,
                                           LayoutB,
                                           LayoutC,
                                           LayoutD>),
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_d,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    auto totalGFlops        = calculateTotalGFlops(m, n, k);
    auto peakGFlopsPerSec   = calculatePeakGFlopsPerSec<InputT, ComputeT, Mi100>(m, n, k, 1087);
    auto actualGFlopsPerSec = calculateGFlopsPerSec(m, n, k, elapsedTimeMs);
    auto efficiency         = actualGFlopsPerSec / peakGFlopsPerSec * 100.0f;

    std::cout << ", " << elapsedTimeMs << ", " << totalGFlops
              << ", " << actualGFlopsPerSec << ", " << efficiency << ", ";

#ifdef WMMA_VALIDATE_TESTS

    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Init reference data and then validate
    std::vector<OutputT> matrixD_ref(m * n, OutputT(0));

    // Give more error tolerance to ComputeT = fp16,
    // due to MFMA output is always fp32. We downcast the MFMA result to fp16, which
    // will introduce an error compared to native fp16 MAC. The tolerance would be a function
    // of max / min values and number of operations propagating the error.
    // Note that integer values between [-2048, 2048 ] are exactly representable by fp16,
    // and significant rounding errors occur thereafter to the nearest multiple of 2.
    // The input generator for GEMM uses integer values within a certain range, therefore
    // FMA operations will be very prone to significant errors.
    double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;

    bool validated = false;
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
    if (!quirks::rocblas_not_support_bf16_bf16_bf16<InputT, OutputT, ComputeT>::value)
    {
        // rocblas matrix C, D always in col_major
        MatrixUtil<col_major>::fill(matrixC, m, n);
        gemm_rocBLAS<InputT, OutputT, ComputeT, LayoutA, LayoutB>(
            m, n, k, matrixA.data(), matrixB.data(), matrixC.data(), matrixD_ref.data(), alpha, beta);

        EXPECT_TRUE( (compareEqual<OutputT, OutputT, LayoutD, col_major>(matrixD, matrixD_ref, m, n, errorTolerance)) );

        //MatrixUtil<LayoutD>::print(matrixD, m, n);
        //MatrixUtil<col_major>::print(matrixD_ref, m, n);

        validated = true;
    }
#endif // WMMA_VALIDATE_WITH_ROCBLAS
    if (!validated)
    {
        gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(
            m, n, k, matrixA.data(), matrixB.data(), matrixC.data(), matrixD_ref.data(), alpha, beta);
        EXPECT_TRUE( (compareEqual<OutputT, OutputT, LayoutD, LayoutD>(matrixD, matrixD_ref, m, n, errorTolerance)) );
    }

#else // WMMA_VALIDATE_TESTS
    // No validation, close off the line.
    std::cout << std::endl;

#endif // WMMA_VALIDATE_TESTS

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT>
inline void test_mma_sync_h(uint32_t TBlockX,
                            uint32_t TBlockY,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K,
                            ComputeT alpha,
                            ComputeT beta)
{
    std::tuple<row_major, col_major> types;
    for_each(types, [&](auto layout_a) {
        for_each(types, [&](auto layout_b) {
            for_each(types, [&](auto layout_c) {
                test_mma_sync_h<BlockM,
                                BlockN,
                                BlockK,
                                InputT,
                                OutputT,
                                ComputeT,
                                decltype(layout_a),
                                decltype(layout_b),
                                decltype(layout_c)>(TBlockX, TBlockY, M, N, K, alpha, beta);
            });
        });
    });
}

template <typename InputT, typename OutputT, typename ComputeT>
inline void test_mma_sync_h_32x32(uint32_t TBlockX, uint32_t TBlockY, uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    // Minimum K = 2 for 32 x 32
    //test_mma_sync_h<32, 32, 2, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    //test_mma_sync_h<32, 32, 4, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<32, 32, 8, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<32, 32, 16, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<32, 32, 32, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<32, 32, 64, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<32, 32, 128, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
}

template <typename InputT, typename OutputT, typename ComputeT>
inline void test_mma_sync_h_16x16(uint32_t TBlockX, uint32_t TBlockY, uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    // Minimum K = 4 for 16 x 16
    //test_mma_sync_h<16, 16, 4, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    //test_mma_sync_h<16, 16, 8, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<16, 16, 16, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<16, 16, 32, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<16, 16, 64, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<16, 16, 128, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    test_mma_sync_h<16, 16, 256, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
    // test_mma_sync_h<16, 16, 512, InputT, OutputT, ComputeT>(TBlockX, TBlockY, M, N, K, alpha, beta);
}

template <typename InputT, typename OutputT, typename ComputeT>
void test_mma_sync_h()
{
    // clang-format off
    std::vector<std::array<int, 2>> thread_block = {{64, 1}, {64, 2}, {64, 4},
                                                    {128,1}, {128,2},
                                                    {256,1}};

    std::vector<std::array<int, 3>> problem_sizes = {{64, 64, 1024},
                                                     {32, 64, 1024},
                                                     {64, 32, 1024},
                                                     {256, 256, 1024},
                                                     {2048, 64, 1024},
                                                     {64, 2048, 1024},
                                                     {1024, 1024, 1024},
                                                     {2048, 2048, 2048},
                                                     {2560, 2560, 2560},
                                                     {3072, 3072, 3072},
                                                     {3584, 3584, 3584},
                                                     {4096, 4096, 4096},
                                                     {5120, 5120, 5120},
                                                     {6144, 6144, 6144},
                                                     {7168, 7168, 7168},
                                                     {8192, 8192, 8192}};

    for(auto tblock : thread_block)
    {
        for(auto size : problem_sizes)
        {
            // skip large sizes when in validation mode
            #ifdef WMMA_VALIDATE_TESTS
            if(size[0] * size[1] > 1024 * 1024)
                continue;
            #endif // WMMA_VALIDATE_TESTS
            auto fargs = std::tuple_cat(tblock, size, std::make_tuple(ComputeT(2.0f), ComputeT(2.0f)));
            std::apply(test_mma_sync_h_16x16<InputT, OutputT, ComputeT>, fargs);
            std::apply(test_mma_sync_h_32x32<InputT, OutputT, ComputeT>, fargs);
        }
    }
    // clang-format on
}

#ifdef WMMA_VALIDATE_TESTS
template <typename... Ts>
void test_mma_sync(std::tuple<Ts...>)
{
    test_mma_sync_h<Ts...>();
}

template <typename T>
struct MmaSyncTest : public testing::Test
{
public:
    void SetUp() override
    {
        headerPrinted = false;
    }
    // TODO: buffer new/del in fixture
};

using Implementations = testing::Types<
    // Native fp16
    std::tuple<float16_t, float16_t, float16_t>,
    std::tuple<float16_t, float16_t, float32_t>,
    std::tuple<float16_t, float32_t, float32_t>,
    // Non-native hfloat16_t (i.e. __half)
    std::tuple<hfloat16_t, hfloat16_t, hfloat16_t>,
    std::tuple<hfloat16_t, hfloat16_t, float32_t>,
    std::tuple<hfloat16_t, float32_t, float32_t>,
    // Non-native bfloat16_t
    std::tuple<bfloat16_t, bfloat16_t, bfloat16_t>,
    std::tuple<bfloat16_t, bfloat16_t, float32_t>,
    std::tuple<bfloat16_t, float32_t, float32_t>,
    // Native fp32
    std::tuple<float32_t, float32_t, float32_t>
>;

TYPED_TEST_SUITE(MmaSyncTest, Implementations);

TYPED_TEST(MmaSyncTest, MmaSync)
{
    TypeParam types;
    test_mma_sync(types);
};

TEST(AdhocMmaSyncTest, AdhocMmaSync)
{
    //test_mma_sync_h<32, 32, 128, float32_t, float32_t>(64, 4, 8192, 8192, 8192, 1.0f, 1.0f);
    //test_mma_sync_h<32, 32, 32, bfloat16_t, bfloat16_t, float32_t, col_major, row_major, row_major>(64, 4, 4096, 4096, 4096, 2.0f, 1.5f);
    test_mma_sync_h<32, 32, 32, float16_t, float16_t, float16_t, col_major, row_major, row_major>(64, 1, 32, 32, 32, 1.0f, 1.0f);
    test_mma_sync_h<32, 32, 32, float16_t, float16_t, float32_t, row_major, row_major, row_major>(64, 1, 64, 64, 128, float16_t(2), float16_t(2));
}

#else // WMMA_VALIDATE_TESTS

int main()
{
    // Native fp16
    test_mma_sync_h<float16_t, float16_t, float16_t>();
    test_mma_sync_h<float16_t, float16_t, float32_t>();
    test_mma_sync_h<float16_t, float32_t, float32_t>();

    // Non-native hfloat16_t (i.e. __half)
    test_mma_sync_h<hfloat16_t, hfloat16_t, hfloat16_t>();
    test_mma_sync_h<hfloat16_t, hfloat16_t, float32_t>();
    test_mma_sync_h<hfloat16_t, float32_t, float32_t>();

    // Non-native bfloat16_t
    test_mma_sync_h<bfloat16_t, bfloat16_t, bfloat16_t>();
    test_mma_sync_h<bfloat16_t, bfloat16_t, float32_t>();
    test_mma_sync_h<bfloat16_t, float32_t, float32_t>();

    // Native fp32
    test_mma_sync_h<float32_t, float32_t, float32_t>();

    //test_mma_sync_h<64, 4, 32, 32, 32, float32_t, float32_t, float32_t>(8192, 8192, 8192, 1.0f, 1.0f);

    //test_mma_sync_h<64, 4, 32, 32, 32, bfloat16_t, bfloat16_t, float32_t, col_major, row_major, row_major>(4096, 4096, 4096, 2.0f, 1.5f);
    //test_mma_sync_h<64, 1, 32, 32, 32, float16_t, float16_t, float16_t, col_major, row_major, row_major>(32, 32, 32, 1.0f, 1.0f);
    //test_mma_sync_h<64, 1, 32, 32, 32, float16_t, float16_t, float32_t, row_major, row_major, row_major>(64, 64, 128, ComputeT(2), ComputeT(2));
    return 0;
}
#endif // WMMA_VALIDATE_TESTS
