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
__global__ void test_load_store_matrix_d(InputT const*   a_in,
                                         InputT const*   b_in,
                                         ComputeT const* c_in,
                                         InputT*         a_out,
                                         InputT*         b_out,
                                         ComputeT*       c_out,
                                         uint32_t        M,
                                         uint32_t        N,
                                         uint32_t        K)
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

    // Map, load and store.
    auto* readA  = MappingA::dataCoord(a_in, lda);
    auto* writeA = MappingA::dataCoord(a_out, lda);
    wmma::load_matrix_sync(fragA, readA, lda);
    wmma::store_matrix_sync(writeA, fragA, lda);

    auto* readB  = MappingB::dataCoord(b_in, ldb);
    auto* writeB = MappingB::dataCoord(b_out, ldb);
    wmma::load_matrix_sync(fragB, readB, ldb);
    wmma::store_matrix_sync(writeB, fragB, ldb);

    auto* readC  = MappingC::dataCoord(c_in, ldc);
    auto* writeC = MappingC::dataCoord(c_out, ldc);
    auto  layoutC
        = std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major : wmma::mem_col_major;
    wmma::load_matrix_sync(fragC, readC, ldc, layoutC);
    wmma::store_matrix_sync(writeC, fragC, ldc, layoutC);
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
__host__ void test_load_store_matrix_h(uint32_t M, uint32_t N, uint32_t K)
{
    std::cout << "HIP wmma::load/store_matrix_sync test: TBlock (" << TBlockX << ", " << TBlockY
              << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N << ", " << K << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "TiTc(" << (std::is_same<InputT, float32_t>::value ? "f32" : "X") << ") \n";

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Initialize input matrices
    std::vector<InputT> matrixA(M * K, 0.0f);
    MatrixUtil<LayoutA>::fill(matrixA, M, K);
    std::vector<InputT> matrixB(K * N, 0.0f);
    MatrixUtil<LayoutB>::fill(matrixB, K, N);
    std::vector<ComputeT> matrixC(M * N, 0.0f);
    MatrixUtil<LayoutC>::fill(matrixC, M, N);

    // Output matrices
    std::vector<InputT>   matrixA_r(M * K, 0.0f);
    std::vector<InputT>   matrixB_r(K * N, 0.0f);
    std::vector<ComputeT> matrixC_r(M * N, 0.0f);

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

    InputT* d_a_r;
    assert(hipMalloc(&d_a_r, bytesA) == hipSuccess);

    InputT* d_b_r;
    assert(hipMalloc(&d_b_r, bytesB) == hipSuccess);

    ComputeT* d_c_r;
    assert(hipMalloc(&d_c_r, bytesC) == hipSuccess);

    auto gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipLaunchKernelGGL((test_load_store_matrix_d<BlockM,
                                                 BlockN,
                                                 BlockK,
                                                 InputT,
                                                 ComputeT,
                                                 LayoutA,
                                                 LayoutB,
                                                 LayoutC>),
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       d_a,
                       d_b,
                       d_c,
                       d_a_r,
                       d_b_r,
                       d_c_r,
                       M,
                       N,
                       K);

    assert(hipMemcpy(matrixA_r.data(), d_a_r, bytesA, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(matrixB_r.data(), d_b_r, bytesB, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(matrixC_r.data(), d_c_r, bytesC, hipMemcpyDeviceToHost) == hipSuccess);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);
    assert(hipFree(d_c) == hipSuccess);
    assert(hipFree(d_a_r) == hipSuccess);
    assert(hipFree(d_b_r) == hipSuccess);
    assert(hipFree(d_c_r) == hipSuccess);

    // Validate
    compareEqual<InputT, InputT, LayoutA, LayoutA>(matrixA, matrixA_r, M, K);
    compareEqual<InputT, InputT, LayoutB, LayoutB>(matrixB, matrixB_r, K, N);
    compareEqual<ComputeT, ComputeT, LayoutC, LayoutC>(matrixC, matrixC_r, M, N);
}

template <uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename ComputeT>
__host__ void test_load_store_matrix_h(uint32_t M, uint32_t N, uint32_t K)
{
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             row_major,
                             row_major,
                             row_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             row_major,
                             col_major,
                             row_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             col_major,
                             row_major,
                             row_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             col_major,
                             col_major,
                             row_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             row_major,
                             row_major,
                             col_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             row_major,
                             col_major,
                             col_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             col_major,
                             row_major,
                             col_major>(M, N, K);
    test_load_store_matrix_h<TBlockX,
                             TBlockY,
                             BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             ComputeT,
                             col_major,
                             col_major,
                             col_major>(M, N, K);
}

void test_load_store_matrix_h()
{
    // Store / load, we must have the same geometry for all matrices.
    // This will exercise matrix a, b and accum load / store layouts.

    // float32_t  64 x 1 threads, block 16 x 16
    test_load_store_matrix_h<64, 1, 16, 16, 16, float32_t, float32_t>(16, 16, 16);
    test_load_store_matrix_h<64, 1, 16, 16, 16, float32_t, float32_t>(32, 32, 32);
    test_load_store_matrix_h<64, 1, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 1, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 1, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 2 threads, block 16 x 16
    test_load_store_matrix_h<64, 2, 16, 16, 16, float32_t, float32_t>(32, 32, 32);
    test_load_store_matrix_h<64, 2, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 2, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 2, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 4 threads, block 16 x 16
    test_load_store_matrix_h<64, 4, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 4, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 4, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 8 threads, block 16 x 16
    test_load_store_matrix_h<64, 8, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 8, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 16 threads, block 16 x 16
    test_load_store_matrix_h<64, 16, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 1 threads, block 16 x 16
    test_load_store_matrix_h<128, 1, 16, 16, 16, float32_t, float32_t>(32, 32, 32);
    test_load_store_matrix_h<128, 1, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<128, 1, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 1, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 2 threads, block 16 x 16
    test_load_store_matrix_h<128, 2, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<128, 2, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 2, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 4 threads, block 16 x 16
    test_load_store_matrix_h<128, 4, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 4, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 8 threads, block 16 x 16
    test_load_store_matrix_h<128, 8, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 1 threads, block 16 x 16
    test_load_store_matrix_h<256, 1, 16, 16, 16, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<256, 1, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<256, 1, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 2 threads, block 16 x 16
    test_load_store_matrix_h<256, 2, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<256, 2, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 4 threads, block 16 x 16
    test_load_store_matrix_h<256, 4, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  512 x 1 threads, block 16 x 16
    test_load_store_matrix_h<512, 1, 16, 16, 16, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<512, 1, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  512 x 2 threads, block 16 x 16
    test_load_store_matrix_h<512, 2, 16, 16, 16, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 1 threads, block 32 x 32
    test_load_store_matrix_h<64, 1, 32, 32, 32, float32_t, float32_t>(32, 32, 32);
    test_load_store_matrix_h<64, 1, 32, 32, 32, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 1, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 1, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 2 threads, block 32 x 32
    test_load_store_matrix_h<64, 2, 32, 32, 32, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 2, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 2, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 4 threads, block 32 x 32
    test_load_store_matrix_h<64, 4, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 4, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 8 threads, block 32 x 32
    test_load_store_matrix_h<64, 8, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 1 threads, block 32 x 32
    test_load_store_matrix_h<128, 1, 32, 32, 32, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<128, 1, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 1, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 2 threads, block 32 x 32
    test_load_store_matrix_h<128, 2, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 2, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 4 threads, block 32 x 32
    test_load_store_matrix_h<128, 4, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 1 threads, block 32 x 32
    test_load_store_matrix_h<256, 1, 32, 32, 32, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<256, 1, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 2 threads, block 32 x 32
    test_load_store_matrix_h<256, 2, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  512 x 1 threads, block 32 x 32
    test_load_store_matrix_h<512, 1, 32, 32, 32, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 1 threads, block 64 x 64
    test_load_store_matrix_h<64, 1, 64, 64, 64, float32_t, float32_t>(64, 64, 64);
    test_load_store_matrix_h<64, 1, 64, 64, 64, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 1, 64, 64, 64, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 2 threads, block 64 x 64
    test_load_store_matrix_h<64, 2, 64, 64, 64, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<64, 2, 64, 64, 64, float32_t, float32_t>(256, 256, 256);

    // float32_t  64 x 4 threads, block 64 x 64
    test_load_store_matrix_h<64, 4, 64, 64, 64, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 1 threads, block 64 x 64
    test_load_store_matrix_h<128, 1, 64, 64, 64, float32_t, float32_t>(128, 128, 128);
    test_load_store_matrix_h<128, 1, 64, 64, 64, float32_t, float32_t>(256, 256, 256);

    // float32_t  128 x 2 threads, block 64 x 64
    test_load_store_matrix_h<128, 2, 64, 64, 64, float32_t, float32_t>(256, 256, 256);

    // float32_t  256 x 1 threads, block 64 x 64
    test_load_store_matrix_h<256, 1, 64, 64, 64, float32_t, float32_t>(256, 256, 256);
}

int main()
{
    test_load_store_matrix_h();
    return 0;
}
