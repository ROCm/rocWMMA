#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"

// LDS capacity helper
template<uint32_t TBlockX,
         uint32_t TBlockY,
         uint32_t BlockM,
         uint32_t BlockN,
         uint32_t BlockK,
         typename InputT,
         typename LayoutA, 
         typename LayoutB>
struct CoopLoadHelper
{
    template<typename MatrixT>
    using DataLayout = typename std::conditional<std::is_same<MatrixT, matrix_a>::value, LayoutA, LayoutB>::type;

    template<typename MatrixT>
    using InputFrag = wmma::fragment<MatrixT, BlockM, BlockN, BlockK, InputT, DataLayout<MatrixT> >;

    template <typename MatrixT>
    using Loader = amdgcn_cooperative_load_dword_DxK<MatrixT, InputFrag<MatrixT>::leadingDim(), InputFrag<MatrixT>::kDim(), InputT, DataLayout<MatrixT>, TBlockY, TBlockX / AMDGCN_WAVE_SIZE>;

    enum : uint32_t 
    {
        LdsUsage = Loader<matrix_a>::Traits::LdsBytes + Loader<matrix_b>::Traits::LdsBytes
    };
};

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK, 
          typename InputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void test_mma_sync_coop_d(const InputT* a,
                                const InputT* b,
                                ComputeT*     c,
                                uint32_t      M,
                                uint32_t      N,
                                uint32_t      K,
                                ComputeT      alpha,
                                ComputeT      beta)
{
    using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, ComputeT, LayoutC>;

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Create frags
    auto fragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>();
    auto fragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>();
    auto fragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>();
    auto fragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>();

    wmma::fill_fragment(fragAcc, 0.0f);

    // Tile using a 2D grid
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / AMDGCN_WAVE_SIZE;

    // Loop over k
    for(int i = 0; i < K; i += BlockK) 
    {
        int aRow = warpM * BlockM;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * BlockN;

        // Bounds checking
        if(aRow < M && aCol < K && bRow < K && bCol < N)
        {
            InputT const* aOffset
                = a
                  + (std::is_same<LayoutA, row_major>::value ? (aRow * lda + aCol)
                                                             : (aRow + aCol * lda));
            InputT const* bOffset
                = b
                  + (std::is_same<LayoutB, row_major>::value ? (bRow * ldb + bCol)
                                                             : (bRow + bCol * ldb));

            // Load the inputs
            wmma::load_matrix_coop_sync(fragA, aOffset, lda); 
            wmma::load_matrix_coop_sync(fragB, bOffset, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * BlockM;
    int cCol = warpN * BlockN;

    if(cRow < M && cCol < N)
    {
        ComputeT* cOffset = c
                            + (std::is_same<LayoutC, row_major>::value ? (cRow * ldc + cCol)
                                                                       : (cRow + cCol * ldc));
        wmma::load_matrix_sync(fragC,
                               cOffset,
                               ldc,
                               std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                       : wmma::mem_col_major);

        for(int i = 0; i < fragC.registerCount(); ++i)
        {
            fragC[i] = alpha * fragAcc[i] + beta * fragC[i];
        }

        // Store the output
        wmma::store_matrix_sync(cOffset,
                                fragC,
                                ldc,
                                std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                        : wmma::mem_col_major);
    }
}

template <uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT,
          typename LayoutA, 
          typename LayoutB,
          typename LayoutC>
__host__ void test_mma_sync_coop_h(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    std::cout << "HIP wmma::mma_sync test: TBlock (" << TBlockX << ", " << TBlockY
              << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N << ", " << K << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "TiTc(" << (std::is_same<InputT, float32_t>::value ? "f32" : "X") << ") \n";

    // Static check on LDS resource requirements.
    // Make sure that we can fit both A and B in LDS if needed.
    static_assert(CoopLoadHelper<TBlockX, TBlockY, BlockM, BlockN, BlockK, InputT, LayoutA, LayoutB>::LdsUsage <= LDS_MAX_BYTES, "Exceeded LDS capacity");

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Initialize input matrices
    std::vector<InputT> matrixA(M * K);
    MatrixUtil<LayoutA>::fill(matrixA, M, K);

    std::vector<InputT> matrixB(K * N);
    MatrixUtil<LayoutB>::fill(matrixB, K, N);

    std::vector<ComputeT> matrixC(M * N, 0.0f);

    // Allocate and copy device memory
    InputT*      d_a;
    const size_t bytesA = matrixA.size() * sizeof(InputT);
    assert(hipMalloc(&d_a, bytesA) == hipSuccess);
    assert(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice) == hipSuccess);

    InputT*      d_b;
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    assert(hipMalloc(&d_b, bytesB) == hipSuccess);
    assert(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice) == hipSuccess);

    ComputeT*    d_c;
    const size_t bytesC = matrixC.size() * sizeof(ComputeT);
    assert(hipMalloc(&d_c, bytesC) == hipSuccess);
    assert(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice) == hipSuccess);

    auto gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipEvent_t startEvent, stopEvent;
    assert(hipEventCreate(&startEvent) == hipSuccess);
    assert(hipEventCreate(&stopEvent) == hipSuccess);
    hipExtLaunchKernelGGL(
        (test_mma_sync_coop_d<BlockM, BlockN, BlockK, InputT, ComputeT, LayoutA, LayoutB, LayoutC>),
        gridDim,
        blockDim,
        LDS_MAX_BYTES, // sharedMemBytes
        0, // stream
        startEvent, // Event start
        stopEvent,  // event stop
        0,          // flags
        d_a,
        d_b,
        d_c,
        M,
        N,
        K,
        alpha,
        beta);

    auto elapsedTimeMs = 0.0f;
    assert(hipEventSynchronize(stopEvent) == hipSuccess);
    assert(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent) == hipSuccess);
    assert(hipEventDestroy(startEvent) == hipSuccess);
    assert(hipEventDestroy(stopEvent) == hipSuccess);

    auto gflops = (2.0f * M * N * K) / 1000000000.0f; 
    std::cout << "Elapsed time (ms): " << elapsedTimeMs << " Speed (Gflops/s): " << gflops / elapsedTimeMs * 1000.0f << std::endl;

    // Copy for validation
    assert(hipMemcpy(matrixC.data(), d_c, bytesC, hipMemcpyDeviceToHost) == hipSuccess);

    // Validate
    std::vector<ComputeT> matrixC_r(M * N, 0.0f);
    gemmCPU<LayoutA, LayoutB, LayoutC, InputT, ComputeT>(
        matrixA,
        matrixB,
        matrixC_r,
        M,
        N,
        K,
        alpha,
        beta);
    compareEqual<InputT, InputT, LayoutA, LayoutA>(matrixC, matrixC_r, M, N);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);
    assert(hipFree(d_c) == hipSuccess);
}

template <uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT>
inline void test_mma_sync_coop_h(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, row_major, row_major, row_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, row_major, col_major, row_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, col_major, row_major, row_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, col_major, col_major, row_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, row_major, row_major, col_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, row_major, col_major, col_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, col_major, row_major, col_major>(M, N, K, alpha, beta);
    test_mma_sync_coop_h<TBlockX, TBlockY, BlockM, BlockN, BlockK, float32_t, float32_t, col_major, col_major, col_major>(M, N, K, alpha, beta);
}

// template<uint32_t TBlockX,
//           uint32_t TBlockY,
//           typename InputT,
//           typename ComputeT>
// inline void test_mma_sync_coop_h_32x32(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
// {
//     // Minimum K = 2 for 32 x 32
//     //test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 2, InputT, ComputeT>(M, N, K, alpha, beta);
//     //test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 4, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 8, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 16, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 32, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 64, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 128, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 256, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 512, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 32, 32, 1024, InputT, ComputeT>(M, N, K, alpha, beta);
// }

// template<uint32_t TBlockX,
//           uint32_t TBlockY,
//           typename InputT,
//           typename ComputeT>
// inline void test_mma_sync_coop_h_16x16(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
// {
//     // Minimum K = 4 for 16 x 16
//     //test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 4, InputT, ComputeT>(M, N, K, alpha, beta);
//     //test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 8, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 16, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 32, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 64, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 128, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 256, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 512, InputT, ComputeT>(M, N, K, alpha, beta);
//     test_mma_sync_coop_h<TBlockX, TBlockY, 16, 16, 1024, InputT, ComputeT>(M, N, K, alpha, beta);
// }

// void test_mma_sync_coop_h()
// {
//     // // float32_t  64 x 1 threads, block 16 x 16 x 4/8/16/32/64/128/256/512/1024,
//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(64, 64, 1024, 2.0f, 2.0f);
//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(32, 64, 1024, 2.0f, 2.0f);
//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(64, 32, 1024, 2.0f, 2.0f);

//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(1024, 2048, 1024, 2.0f, 2.0f);
//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(2048, 64, 1024, 2.0f, 2.0f);
//     // test_mma_sync_coop_h_16x16<64, 1, float32_t, float32_t>(2048, 2048, 1024, 2.0f, 2.0f);

//     // float32_t  64 x 1 threads, block 32 x 32 x 2/4/8/16/32/64/128/256/512/1024,
//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(64, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(32, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(64, 32, 1024, 2.0f, 2.0f);

//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(1024, 2048, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(2048, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 1, float32_t, float32_t>(2048, 2048, 1024, 2.0f, 2.0f);

//     // float32_t  64 x 2 threads, block 32 x 32 x 2/4/8/16/32/64/128/256/512/1024,
//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(64, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(128, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(64, 128, 1024, 2.0f, 2.0f);

//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(1024, 2048, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(2048, 64, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 2, float32_t, float32_t>(2048, 2048, 1024, 2.0f, 2.0f);

//     // float32_t  64 x 4 threads, block 32 x 32 x 2/4/8/16/32/64/128/256/512/1024,
//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(128, 128, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(128, 256, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(256, 128, 1024, 2.0f, 2.0f);

//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(1024, 2048, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(2048, 128, 1024, 2.0f, 2.0f);
//     test_mma_sync_coop_h_32x32<64, 4, float32_t, float32_t>(2048, 2048, 1024, 2.0f, 2.0f);
// }

int main()
{
    //test_mma_sync_coop_h(); 
    //test_mma_sync_coop_h<64, 1, 16, 16, 8, float32_t, float32_t>(4096, 4096, 256, 2.0f, 2.0f);
    //test_mma_sync_coop_h<64, 1, 16, 16, 32, float32_t, float32_t>(4096, 4096, 256, 2.0f, 2.0f); 
    test_mma_sync_coop_h<256, 2 , 32, 32, 64, float32_t, float32_t>(4096, 4096, 4096, 2.0f, 2.0f);
    return 0;
}
