#include <hip/hip_runtime.h>

#include <unistd.h>
#include <type_traits>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"
#include <gtest/gtest.h>

#include "Common.hpp"

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
    if ( M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK )
        return;

    std::cout << "HIP wmma::fill_fragment test: TBlock (" << TBlockX << ", " << TBlockY << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N << ", " << K << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "TiTc(" << dataTypeToString<InputT>() << "_" << dataTypeToString<ComputeT>()
              << ") \n";

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
    EXPECT_TRUE( (compareEqual<InputT, InputT, LayoutA, LayoutA>(matrixA, refA, M, K)) );
    EXPECT_TRUE( (compareEqual<InputT, InputT, LayoutB, LayoutB>(matrixB, refB, K, N)) );
    EXPECT_TRUE( (compareEqual<ComputeT, ComputeT, LayoutC, LayoutC>(matrixC, refC, M, N)) );
}

template <typename T>
struct FillFragmentTest : public testing::Test
{
    // TODO: buffer new/del in fixture
};

template <typename IntConstBlockM,
          typename IntConstBlockN,
          typename IntConstBlockK,
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
    std::tuple<row_major, col_major> types;
    for_each(types, [&](auto layout_a) {
        for_each(types, [&](auto layout_b) {
            for_each(types, [&](auto layout_c) {
                test_fill_fragment_h<IntConstBlockM::value,
                                     IntConstBlockN::value,
                                     IntConstBlockK::value,
                                     InputT,
                                     ComputeT,
                                     decltype(layout_a),
                                     decltype(layout_b),
                                     decltype(layout_c)>(
                    TBlockX, TBlockY, M, N, K, fillA, fillB, fillC);
            });
        });
    });
}

template <typename... Ts>
void test_fill_fragment(std::tuple<Ts...>)
{
    // clang-format off
    std::vector<std::array<int, 2>> thread_block = {{64, 1}, {64, 2}, {64, 4}, {64, 8}, {64, 16},
                                                    {128,1}, {128,2}, {128,4}, {128,8},
                                                    {256,1}, {256,2}, {256,4},
                                                    {512,1}, {512,2}};

    // For fills, we must have the same geometry for all matrices
    std::vector<std::array<int, 3>> problem_sizes = {{16, 16, 16},
                                                     {32, 32, 32},
                                                     {64, 64, 64},
                                                     {128, 128, 128},
                                                     {256, 256, 256},
                                                     {2048, 2048, 2048}};
    // clang-format on
    for(auto tblock : thread_block)
    {
        for (auto size : problem_sizes)
        {
            auto fargs = std::tuple_cat(tblock, size, std::make_tuple(1.0f, 2.0f, -3.0f));
            std::apply(test_fill_fragment_h<Ts...>, fargs);
        }
    }
}

using Implementations = testing::Types<
    // BlockM, BlockN, BlockK, InputT, ComputeT
    std::tuple<I<16>, I<16>, I<16>, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<32>, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<16>, float16_t, float16_t>,
    std::tuple<I<32>, I<32>, I<32>, float16_t, float16_t>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, hfloat16_t>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t, hfloat16_t>
>;

TYPED_TEST_SUITE(FillFragmentTest, Implementations);

TYPED_TEST(FillFragmentTest, FillFragment)
{
    TypeParam types;
    test_fill_fragment(types);
};
