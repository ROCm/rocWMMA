#include <hip/hip_runtime.h>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void test_mma_sync_d(const InputT* a,
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

    int iterations = K / BlockK;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / AMDGCN_WAVE_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

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
            wmma::load_matrix_sync(fragA, aOffset, lda);
            wmma::load_matrix_sync(fragB, bOffset, ldb);

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
                               c + cRow + cCol * ldc,
                               ldc,
                               std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                       : wmma::mem_col_major);

        for(int i = 0; i < fragC.num_elements(); i++)
        {
            fragC[i] = alpha * fragAcc[i] + beta * fragC[i];
        }

        // Store the output
        wmma::store_matrix_sync(cOffset,
                                fragAcc,
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
__host__ void test_mma_sync_h(uint32_t M, uint32_t N, uint32_t K, ComputeT alpha, ComputeT beta)
{
    std::cout << "HIP matrix mult test\n";

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

    hipLaunchKernelGGL(
        (test_mma_sync_d<BlockM, BlockN, BlockK, InputT, ComputeT, LayoutA, LayoutB, LayoutC>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        d_a,
        d_b,
        d_c,
        M,
        N,
        K,
        alpha,
        beta);

    assert(hipMemcpy(matrixC.data(), d_c, bytesC, hipMemcpyDeviceToHost) == hipSuccess);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);
    assert(hipFree(d_c) == hipSuccess);
}

void test_mma_sync_h()
{
    // For fills, we must have the same geometry for all matrices

    // float32_t  64 x 1 threads, block 64 x 64,

    // MNK = 64
    test_mma_sync_h<64, 1, 32, 32, 32, float32_t, float32_t, row_major, row_major, row_major>(
        64, 64, 64, 1.0f, 0.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (64, 64, 64, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (64, 64, 64, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (64, 64, 64, 10.0f, -11.0f, 12.0f);

    // // MNK = 128
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (128, 128, 128, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (128, 128, 128, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (128, 128, 128, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (128, 128, 128, 10.0f, -11.0f, 12.0f);

    // // MNK = 512
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // MNK = 16384
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (16384, 16384, 16384, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (16384, 16384, 16384, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (16384, 16384, 16384, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (16384, 16384, 16384, 10.0f, -11.0f, 12.0f);

    // // float32_t  64 x 2 threads, block 64 x 64,

    // // MNK = 128
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (128, 128, 128, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (128, 128, 128, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (128, 128, 128, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (128, 128, 128, 10.0f, -11.0f, 12.0f);

    // // MNK = 512
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // MNK = 16384
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (16384, 16384, 16384, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (16384, 16384, 16384, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (16384, 16384, 16384, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (16384, 16384, 16384, 10.0f, -11.0f, 12.0f);

    // // float32_t  64 x 4 threads, block 64 x 64,

    // // MNK = 512
    // test_fill_fragment_h<64, 4, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 4, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 4, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 4, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  64 x 8 threads, block 64 x 64,

    // // MNK = 512
    // test_fill_fragment_h<64, 8, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<64, 8, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<64, 8, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<64, 8, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  128 x 1 threads, block 64 x 64,

    // // MNK = 128
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (128, 128, 128, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (128, 128, 128, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (128, 128, 128, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (128, 128, 128, 10.0f, -11.0f, 12.0f);

    // // MNK = 512
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<128, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  128 x 2 threads, block 64 x 64,

    // // MNK = 128
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (128, 128, 128, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (128, 128, 128, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (128, 128, 128, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (128, 128, 128, 10.0f, -11.0f, 12.0f);

    // // MNK = 512
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<128, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  128 x 4 threads, block 64 x 64,

    // // MNK = 512
    // test_fill_fragment_h<128, 4, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<128, 4, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<128, 4, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<128, 4, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  256 x 1 threads, block 64 x 64,

    // // MNK = 512
    // test_fill_fragment_h<256, 1, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<256, 1, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<256, 1, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<256, 1, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);

    // // float32_t  256 x 2 threads, block 64 x 64,

    // // MNK = 512
    // test_fill_fragment_h<256, 2, 64, 64, 64, float32_t, float32_t, row_major, row_major, row_major>
    // (512, 512, 512, -1.0f, 2.0f, -3.0f);
    // test_fill_fragment_h<256, 2, 64, 64, 64, float32_t, float32_t, row_major, col_major, row_major>
    // (512, 512, 512, 4.0f, -5.0f, 6.0f);
    // test_fill_fragment_h<256, 2, 64, 64, 64, float32_t, float32_t, col_major, row_major, row_major>
    // (512, 512, 512, -7.0f, 8.0f, -9.0f);
    // test_fill_fragment_h<256, 2, 64, 64, 64, float32_t, float32_t, col_major, col_major, row_major>
    // (512, 512, 512, 10.0f, -11.0f, 12.0f);
}

int main()
{
    test_mma_sync_h();
    return 0;
}
