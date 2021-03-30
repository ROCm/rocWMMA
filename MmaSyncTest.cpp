#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "Performance.h"
#include "Utils.h"
#include "WMMA.h"

#ifdef WMMA_VALIDATE_TESTS
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
#include "rocBLASReference.h" // rocBLAS GPU kernel
#else
#include "Reference.h" // Vanilla CPU kernel
#endif // WMMA_VALIDATE_WITH_ROCBLAS
#endif // WMMA_VALIDATE_TESTS

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
__global__ void test_mma_sync_d(uint32_t       m,
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

template <uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void test_mma_sync_h(uint32_t m, uint32_t n, uint32_t k, ComputeT alpha, ComputeT beta)
{
    int lda = std::is_same<LayoutA, row_major>::value ? k : m;
    int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
    int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
    int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n, 0.0f);
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

    assert(hipMalloc(&d_a, bytesA) == hipSuccess);
    assert(hipMalloc(&d_b, bytesB) == hipSuccess);
    assert(hipMalloc(&d_c, bytesC) == hipSuccess);
    assert(hipMalloc(&d_d, bytesD) == hipSuccess);

    assert(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice) == hipSuccess);

    auto gridDim
        = dim3(ceilDiv(m, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(n, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipEvent_t startEvent, stopEvent;
    assert(hipEventCreate(&startEvent) == hipSuccess);
    assert(hipEventCreate(&stopEvent) == hipSuccess);

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
                          max(BlockM * blockDim.y, BlockN * blockDim.x / AMDGCN_WAVE_SIZE) * BlockK
                              * sizeof(InputT), // sharedMemBytes
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
    assert(hipEventSynchronize(stopEvent) == hipSuccess);
    assert(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent) == hipSuccess);
    assert(hipEventDestroy(startEvent) == hipSuccess);
    assert(hipEventDestroy(stopEvent) == hipSuccess);

    auto totalGFlops        = calculateTotalGFlops(m, n, k);
    auto peakGFlopsPerSec   = calculatePeakGFlopsPerSec<InputT, ComputeT, Mi100>(m, n, k, 1087);
    auto actualGFlopsPerSec = calculateGFlopsPerSec(m, n, k, elapsedTimeMs);
    auto efficiency         = actualGFlopsPerSec / peakGFlopsPerSec * 100.0f;

    std::cout << "TBlkX, TBlkY, BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, "
                 "ldd, LytA_LytB_LytC_LytD, Ti_To_Tc, elapsedMs, GFlops, GFlops/s, Efficiency(%) = "
              << TBlockX << ", " << TBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
              << ", " << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
              << ", " << beta << ", " << ldc << ", " << ldd << ", "
              << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutD, row_major>::value ? "R" : "C") << ", "
              << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
              << dataTypeToString<ComputeT>() << ", " << elapsedTimeMs << ", " << totalGFlops
              << ", " << actualGFlopsPerSec << ", " << efficiency << ", ";

#ifdef WMMA_VALIDATE_TESTS

    assert(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost) == hipSuccess);

    // Init reference data and then validate
    std::vector<OutputT> matrixD_ref(m * n, 0.0f);

#ifdef WMMA_VALIDATE_WITH_ROCBLAS

    // rocblas matrix C, D always in col_major
    MatrixUtil<col_major>::fill(matrixC, m, n);
    gemm_rocBLAS<InputT, OutputT, ComputeT, LayoutA, LayoutB>(
        m, n, k, matrixA.data(), matrixB.data(), matrixC.data(), matrixD_ref.data(), alpha, beta);
    compareEqual<OutputT, OutputT, LayoutD, col_major>(matrixD, matrixD_ref, m, n);

    //MatrixUtil<LayoutD>::print(matrixD, m, n);
    //MatrixUtil<col_major>::print(matrixD_ref, m, n);

#else

    gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(
        m, n, k, matrixA.data(), matrixB.data(), matrixC.data(), matrixD_ref.data(), alpha, beta);
    compareEqual<OutputT, OutputT, LayoutD, LayoutD>(matrixD, matrixD_ref, m, n);

#endif // WMMA_VALIDATE_WITH_ROCBLAS

#else
    // No validation, close off the line.
    std::cout << std::endl;

#endif // WMMA_VALIDATE_TESTS

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);
    assert(hipFree(d_c) == hipSuccess);
    assert(hipFree(d_d) == hipSuccess);
}

template <uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT>
inline void test_mma_sync_h(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    row_major,
                    row_major,
                    row_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    row_major,
                    col_major,
                    row_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    col_major,
                    row_major,
                    row_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    col_major,
                    col_major,
                    row_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    row_major,
                    row_major,
                    col_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    row_major,
                    col_major,
                    col_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    col_major,
                    row_major,
                    col_major>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX,
                    TBlockY,
                    BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    col_major,
                    col_major,
                    col_major>(M, N, K, alpha, beta);
}

template <uint32_t TBlockX, uint32_t TBlockY, typename InputT, typename OutputT, typename ComputeT>
inline void test_mma_sync_h_32x32(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    // Minimum K = 2 for 32 x 32
    //test_mma_sync_h<TBlockX, TBlockY, 32, 32, 2, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    //test_mma_sync_h<TBlockX, TBlockY, 32, 32, 4, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 32, 32, 8, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 32, 32, 16, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 32, 32, 32, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 32, 32, 64, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 32, 32, 128, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
}

template <uint32_t TBlockX, uint32_t TBlockY, typename InputT, typename OutputT, typename ComputeT>
inline void test_mma_sync_h_16x16(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    // Minimum K = 4 for 16 x 16
    //test_mma_sync_h<TBlockX, TBlockY, 16, 16, 4, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    //test_mma_sync_h<TBlockX, TBlockY, 16, 16, 8, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 16, 16, 16, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 16, 16, 32, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 16, 16, 64, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 16, 16, 128, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    test_mma_sync_h<TBlockX, TBlockY, 16, 16, 256, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
    // test_mma_sync_h<TBlockX, TBlockY, 16, 16, 512, InputT, OutputT, ComputeT>(M, N, K, alpha, beta);
}

template <typename InputT, typename OutputT, typename ComputeT>
void test_mma_sync_h()
{
    // 16 x 16
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(64, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(32, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(64, 32, 1024, 2.0f, 2.0f);

    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 2, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(2048, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<64, 4, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<128, 2, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(2048, 1024, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_16x16<256, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    // 32 x 32
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(64, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(32, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(64, 32, 1024, 2.0f, 2.0f);

    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(64, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(128, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(64, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 2, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(128, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(128, 256, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(256, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(2048, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<64, 4, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(64, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(128, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(64, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(64, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(128, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(64, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(2048, 64, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<128, 2, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);

    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(128, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(128, 256, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(256, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(1024, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(2048, 128, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(2048, 2048, 1024, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(2048, 2048, 2048, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(2560, 2560, 2560, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(3072, 3072, 3072, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(3584, 3584, 3584, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(4096, 4096, 4096, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(5120, 5120, 5120, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(6144, 6144, 6144, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(7168, 7168, 7168, 2.0f, 2.0f);
    test_mma_sync_h_32x32<256, 1, InputT, OutputT, ComputeT>(8192, 8192, 8192, 2.0f, 2.0f);
}

int main()
{
    test_mma_sync_h<float16_t, float16_t, float32_t>();
    //test_mma_sync_h<float32_t, float32_t>();

    //test_mma_sync_h<64, 4, 32, 32, 128, float32_t, float32_t>(8192, 8192, 8192, 1.0f, 1.0f);

    //test_mma_sync_h<64, 2, 16, 16, 64, float16_t, float32_t, row_major, row_major, col_major>(1024, 2048, 1024, 1.0f, 1.0f);
    //test_mma_sync_h<64, 1, 16, 16, 16, float16_t, float16_t, float32_t, row_major, row_major, row_major>(64, 64, 128, 2.0f, 2.0f);
    return 0;
}
