#include <hip/hip_runtime.h>

#include <unistd.h>

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
__global__ void test_fill_fragment_d(InputT*   a,
                                     InputT*   b,
                                     ComputeT* c,
                                     uint32_t  M,
                                     uint32_t  N,
                                     uint32_t  K,
                                     InputT    fillA,
                                     InputT    fillB,
                                     ComputeT  fillC)
{
    using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, ComputeT, LayoutC>;

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Create frags and fill
    auto fragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>();
    auto fragB = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>();
    auto fragC = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>();

    wmma::fill_fragment(fragA, fillA);
    wmma::fill_fragment(fragB, fillB);
    wmma::fill_fragment(fragC, fillC);

    // Map and store
    auto* offsetA = MappingA::dataCoord(a, lda);
    wmma::store_matrix_sync(offsetA, fragA, lda);

    auto* offsetB = MappingB::dataCoord(b, ldb);
    wmma::store_matrix_sync(offsetB, fragB, ldb);

    auto* offsetC = MappingC::dataCoord(c, ldc);
    wmma::store_matrix_sync(offsetC,
                            fragC,
                            ldc,
                            std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                    : wmma::mem_col_major);
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__host__ void test_fill_fragment_h(uint32_t TBlockX,
                                   uint32_t TBlockY,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   InputT   fillA,
                                   InputT   fillB,
                                   ComputeT fillC)
{
    std::cout << "HIP wmma::fill_fragment test: TBlock (" << TBlockX << ", " << TBlockY << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N << ", " << K << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "TiTc(" << dataTypeToString<InputT>() << "_" << dataTypeToString<ComputeT>() << ") \n";

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Initialize input matrices
    std::vector<InputT>   matrixA(M * K, 0.0f);
    std::vector<InputT>   matrixB(K * N, 0.0f);
    std::vector<ComputeT> matrixC(M * N, 0.0f);

    // Allocate and copy init values to device memory
    InputT*      d_a;
    const size_t bytesA = matrixA.size() * sizeof(InputT);
    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));

    InputT*      d_b;
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));

    ComputeT*    d_c;
    const size_t bytesC = matrixC.size() * sizeof(ComputeT);
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));

    auto gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipLaunchKernelGGL(
        (test_fill_fragment_d<BlockM, BlockN, BlockK, InputT, ComputeT, LayoutA, LayoutB, LayoutC>),
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
        fillA,
        fillB,
        fillC);

    CHECK_HIP_ERROR(hipMemcpy(matrixA.data(), d_a, bytesA, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(matrixB.data(), d_b, bytesB, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(matrixC.data(), d_c, bytesC, hipMemcpyDeviceToHost));

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));

    // Initialize reference matrices
    std::vector<InputT>   refA(M * K, fillA);
    std::vector<InputT>   refB(K * N, fillB);
    std::vector<ComputeT> refC(M * N, fillC);

    // Compare
    compareEqual<InputT, InputT, LayoutA, LayoutA>(matrixA, refA, M, K);
    compareEqual<InputT, InputT, LayoutB, LayoutB>(matrixB, refB, K, N);
    compareEqual<ComputeT, ComputeT, LayoutC, LayoutC>(matrixC, refC, M, N);
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT>
__host__ void test_fill_fragment_h(uint32_t TBlockX,
                                   uint32_t TBlockY,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   InputT   fillA,
                                   InputT   fillB,
                                   ComputeT fillC)
{
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         row_major,
                         row_major,
                         row_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         row_major,
                         col_major,
                         row_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         col_major,
                         row_major,
                         row_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         col_major,
                         col_major,
                         row_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         row_major,
                         row_major,
                         col_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         row_major,
                         col_major,
                         col_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         col_major,
                         row_major,
                         col_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
    test_fill_fragment_h<BlockM,
                         BlockN,
                         BlockK,
                         InputT,
                         ComputeT,
                         col_major,
                         col_major,
                         col_major>(TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
}

template<typename InputT, typename ComputeT>
void test_fill_fragment_h()
{
    // For fills, we must have the same geometry for all matrices

    // float32_t  64 x 1 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 1, 16, 16, 16, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 1, 32, 32, 32, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        64, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 2 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 2, 32, 32, 32, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 2, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        64, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 4 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 4, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 4, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        64, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 8 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 8, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 8, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        64, 8, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 16 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(64, 16, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        64, 16, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 1 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 1, 32, 32, 32, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        128, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 2 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 2, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        128, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 4 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 4, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        128, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 8 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(128, 8, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        128, 8, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 1 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        256, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 2 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        256, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 4 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(256, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        256, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  512 x 1 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(512, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(512, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        512, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  512 x 2 threads, block 16 x 16
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(512, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<16, 16, 16, InputT, ComputeT>(
        512, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 1 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 1, 32, 32, 32, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        64, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 2 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 2, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        64, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 4 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 4, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        64, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 8 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(64, 8, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        64, 8, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 1 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        128, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 2 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        128, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 4 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(128, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        128, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 1 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(256, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(256, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        256, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 2 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(256, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        256, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  512 x 1 threads, block 32 x 32
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(512, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<32, 32, 32, InputT, ComputeT>(
        512, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 1 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 1, 64, 64, 64, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        64, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 2 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 2, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        64, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  64 x 4 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(64, 4, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        64, 4, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 1 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(128, 1, 128, 128, 128, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(128, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        128, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  128 x 2 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(128, 2, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        128, 2, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);

    // float32_t  256 x 1 threads, block 64 x 64
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(256, 1, 256, 256, 256, -1.0f, 2.0f, -3.0f);
    test_fill_fragment_h<64, 64, 64, InputT, ComputeT>(
        256, 1, 16384, 16384, 16384, -1.0f, 2.0f, -3.0f);
}

int main()
{
    test_fill_fragment_h<float16_t, float16_t>();
    test_fill_fragment_h<float16_t, float32_t>();
    test_fill_fragment_h<float32_t, float32_t>();
    return 0;
}
