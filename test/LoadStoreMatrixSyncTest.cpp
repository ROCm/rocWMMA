#include <hip/hip_runtime.h>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "MappingUtil.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"

#include <gtest/gtest.h>
#include "Common.hpp"

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename DataT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void test_load_store_matrix_d(DataT const* a_in,
                                         DataT const* b_in,
                                         DataT const* c_in,
                                         DataT*       a_out,
                                         DataT*       b_out,
                                         DataT*       c_out,
                                         uint32_t     M,
                                         uint32_t     N,
                                         uint32_t     K)
{
    using MappingA = MappingUtil<BlockM, BlockN, DataT, LayoutA>;
    using MappingB = MappingUtil<BlockM, BlockN, DataT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, DataT, LayoutC>;

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Create frags and fill
    auto fragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>();
    auto fragB = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>();
    auto fragC = wmma::fragment<accumulator, BlockM, BlockN, BlockK, DataT>();

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

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename DataT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__host__ void test_load_store_matrix_h(uint32_t TBlockX, uint32_t TBlockY, uint32_t M, uint32_t N, uint32_t K)
{
    if ( M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK )
        return;

    std::cout << "HIP wmma::load/store_matrix_sync test: TBlock (" << TBlockX << ", " << TBlockY
              << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N <<  ", " << K  << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "T(" << dataTypeToString<DataT>() << ") \n";

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Initialize input matrices
    std::vector<DataT> matrixA(M * K, DataT(0));
    MatrixUtil<LayoutA>::fill(matrixA, M, K);
    std::vector<DataT> matrixB(K * N, DataT(0));
    MatrixUtil<LayoutB>::fill(matrixB, K, N);
    std::vector<DataT> matrixC(M * N, DataT(0));
    MatrixUtil<LayoutC>::fill(matrixC, M, N);

    // Output matrices
    std::vector<DataT> matrixA_r(M * K, DataT(0));
    std::vector<DataT> matrixB_r(K * N, DataT(0));
    std::vector<DataT> matrixC_r(M * N, DataT(0));

    // Allocate and copy device memory
    DataT*       d_a;
    const size_t bytesA = matrixA.size() * sizeof(DataT);
    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));

    DataT*       d_b;
    const size_t bytesB = matrixB.size() * sizeof(DataT);
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));

    DataT*       d_c;
    const size_t bytesC = matrixC.size() * sizeof(DataT);
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));

    DataT* d_a_r;
    CHECK_HIP_ERROR(hipMalloc(&d_a_r, bytesA));

    DataT* d_b_r;
    CHECK_HIP_ERROR(hipMalloc(&d_b_r, bytesB));

    DataT* d_c_r;
    CHECK_HIP_ERROR(hipMalloc(&d_c_r, bytesC));

    auto gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    std::cout << "Grid Dim: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block Dim: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;

    hipLaunchKernelGGL(
        (test_load_store_matrix_d<BlockM, BlockN, BlockK, DataT, LayoutA, LayoutB, LayoutC>),
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

    CHECK_HIP_ERROR(hipMemcpy(matrixA_r.data(), d_a_r, bytesA, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(matrixB_r.data(), d_b_r, bytesB, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(matrixC_r.data(), d_c_r, bytesC, hipMemcpyDeviceToHost));

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_a_r));
    CHECK_HIP_ERROR(hipFree(d_b_r));
    CHECK_HIP_ERROR(hipFree(d_c_r));

    // Validate
    EXPECT_TRUE( (compareEqual<DataT, DataT, LayoutA, LayoutA>(matrixA, matrixA_r, M, K)) );
    //MatrixUtil<LayoutC>::print(matrixA_r, M, N);
    EXPECT_TRUE( (compareEqual<DataT, DataT, LayoutB, LayoutB>(matrixB, matrixB_r, K, N)) );
    EXPECT_TRUE( (compareEqual<DataT, DataT, LayoutC, LayoutC>(matrixC, matrixC_r, M, N)) );
}

template <typename IntConstBlockM,
          typename IntConstBlockN,
          typename IntConstBlockK,
          typename DataT>
void test_load_store_matrix_h(uint32_t TBlockX, uint32_t TBlockY, uint32_t M, uint32_t N, uint32_t K)
{
    std::tuple<row_major, col_major> types;
    for_each(types, [&](auto layout_a) {
        for_each(types, [&](auto layout_b) {
            for_each(types, [&](auto layout_c) {
                test_load_store_matrix_h<IntConstBlockM::value,
                                         IntConstBlockN::value,
                                         IntConstBlockK::value,
                                         DataT,
                                         decltype(layout_a),
                                         decltype(layout_b),
                                         decltype(layout_c)>(TBlockX, TBlockY, M, N, K);
            });
        });
    });
}

template <typename... Ts>
void test_load_store_matrix(std::tuple<Ts...>)
{
    // clang-format off
    std::vector<std::array<int, 2>> thread_block = {{64, 1}, {64, 2}, {64, 4}, {64, 8}, {64, 16},
                                                    {128,1}, {128,2}, {128,4}, {128,8},
                                                    {256,1}, {256,2}, {256,4},
                                                    {512,1}, {512,2}};
    std::vector<std::array<int, 3>> problem_sizes = {{16, 16, 16},
                                                     {32, 32, 32},
                                                     {64, 64, 64},
                                                     {128, 128, 128},
                                                     {256, 256, 256}};
                                                    //  {2048, 256, 512}}; // TODO: check why this fails
    // clang-format on
    for(auto tblock : thread_block)
    {
        for (auto size : problem_sizes)
        {
            auto fargs = std::tuple_cat(tblock, size);
            std::apply(test_load_store_matrix_h<Ts...>, fargs);
        }
    }
}

template <typename T>
struct LoadStoreMatrixSyncTest : public testing::Test
{
    // TODO: buffer new/del in fixture
};

using Implementations = testing::Types<
    // BlockM, BlockN, BlockK, InputT/ComputeT
    std::tuple<I<16>, I<16>, I<16>, float32_t>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t>,
    std::tuple<I<16>, I<16>, I<16>, float16_t>,
    std::tuple<I<16>, I<16>, I<16>, bfloat16_t>,

    std::tuple<I<32>, I<32>, I<32>, float32_t>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t>,
    std::tuple<I<32>, I<32>, I<32>, float16_t>,
    std::tuple<I<32>, I<32>, I<32>, bfloat16_t>,

    std::tuple<I<64>, I<64>, I<64>, float32_t>,
    std::tuple<I<64>, I<64>, I<64>, hfloat16_t>,
    std::tuple<I<64>, I<64>, I<64>, float16_t>,
    std::tuple<I<64>, I<64>, I<64>, bfloat16_t>
>;

TYPED_TEST_SUITE(LoadStoreMatrixSyncTest, Implementations);

TYPED_TEST(LoadStoreMatrixSyncTest, LoadStoreMatrixSync)
{
    TypeParam types;
    test_load_store_matrix(types);
};
