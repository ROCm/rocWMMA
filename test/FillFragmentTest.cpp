#include <hip/hip_runtime.h>

#include <type_traits>
#include <unistd.h>

#include "Constants.h"
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
__global__ void test_fill_fragment_d(DataT*   a,
                                     DataT*   b,
                                     DataT*   c,
                                     uint32_t M,
                                     uint32_t N,
                                     uint32_t K,
                                     DataT    fillA,
                                     DataT    fillB,
                                     DataT    fillC)
{
    using MappingA = MappingUtil<BlockM, BlockK, DataT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, DataT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, DataT, LayoutC>;

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Create frags and fill
    auto fragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>();
    auto fragB = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>();
    auto fragC = wmma::fragment<accumulator, BlockM, BlockN, BlockK, DataT>();

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
          typename DataT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__host__ void test_fill_fragment_h(uint32_t TBlockX,
                                   uint32_t TBlockY,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   DataT    fillA,
                                   DataT    fillB,
                                   DataT    fillC)
{
    if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK)
        return;

    std::cout << "HIP wmma::fill_fragment test: TBlock (" << TBlockX << ", " << TBlockY << ") "
              << "BlockMNK(" << BlockM << ", " << BlockN << ", " << BlockK << ") "
              << "MatrixMNK(" << M << ", " << N << ", " << K << ") "
              << "FmtABC(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << ", "
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << ") "
              << "T(" << dataTypeToString<DataT>() << ") \n";

    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    // Initialize input matrices
    std::vector<DataT> matrixA(M * K, DataT(0));
    std::vector<DataT> matrixB(K * N, DataT(0));
    std::vector<DataT> matrixC(M * N, DataT(0));

    // Allocate and copy init values to device memory
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

    auto gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    hipLaunchKernelGGL(
        (test_fill_fragment_d<BlockM, BlockN, BlockK, DataT, LayoutA, LayoutB, LayoutC>),
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
    std::vector<DataT> refA(M * K, fillA);
    std::vector<DataT> refB(K * N, fillB);
    std::vector<DataT> refC(M * N, fillC);

    // Compare
    EXPECT_TRUE((compareEqual<DataT, DataT, LayoutA, LayoutA>(matrixA, refA, M, K)));
    EXPECT_TRUE((compareEqual<DataT, DataT, LayoutB, LayoutB>(matrixB, refB, K, N)));
    EXPECT_TRUE((compareEqual<DataT, DataT, LayoutC, LayoutC>(matrixC, refC, M, N)));
}

template <typename T>
struct FillFragmentTest : public testing::Test
{
    // TODO: buffer new/del in fixture
};

template <typename IntConstBlockM, typename IntConstBlockN, typename IntConstBlockK, typename DataT>
__host__ void test_fill_fragment_h(uint32_t TBlockX,
                                   uint32_t TBlockY,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   DataT    fillA,
                                   DataT    fillB,
                                   DataT    fillC)
{
    std::tuple<row_major, col_major> types;
    for_each(types, [&](auto layout_a) {
        for_each(types, [&](auto layout_b) {
            for_each(types, [&](auto layout_c) {
                test_fill_fragment_h<IntConstBlockM::value,
                                     IntConstBlockN::value,
                                     IntConstBlockK::value,
                                     DataT,
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
                                                     {1024, 1024, 1024},
                                                     {2048, 2048, 2048}};
    // clang-format on
    for(auto tblock : thread_block)
    {
        for(auto size : problem_sizes)
        {
            // Incoming Ts... args are BlockM, BlockN, BlockK, DataT.
            using DataT = typename std::tuple_element<3, std::tuple<Ts...>>::type;
            auto fargs  = std::tuple_cat(
                 tblock, size, std::make_tuple(DataT(99.2), DataT(2), DataT(-3.333333)));
            std::apply(test_fill_fragment_h<Ts...>, fargs);
        }
    }
}

using Implementations = testing::Types<
    // BlockM, BlockN, BlockK, InputT, ComputeT
    std::tuple<I<16>, I<16>, I<16>, int8_t>,
    std::tuple<I<16>, I<16>, I<16>, uint8_t>,
    std::tuple<I<16>, I<16>, I<16>, int32_t>,
    std::tuple<I<16>, I<16>, I<16>, uint32_t>,
    std::tuple<I<16>, I<16>, I<16>, bfloat16_t>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t>,
    std::tuple<I<16>, I<16>, I<16>, float16_t>,
    std::tuple<I<16>, I<16>, I<16>, float32_t>,
    std::tuple<I<16>, I<16>, I<16>, float64_t>,

    std::tuple<I<32>, I<32>, I<32>, int8_t>,
    std::tuple<I<32>, I<32>, I<32>, uint8_t>,
    std::tuple<I<32>, I<32>, I<32>, int32_t>,
    std::tuple<I<32>, I<32>, I<32>, uint32_t>,
    std::tuple<I<32>, I<32>, I<32>, bfloat16_t>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t>,
    std::tuple<I<32>, I<32>, I<32>, float16_t>,
    std::tuple<I<32>, I<32>, I<32>, float32_t>,
    std::tuple<I<32>, I<32>, I<32>, float64_t>,

    std::tuple<I<64>, I<64>, I<64>, int8_t>,
    std::tuple<I<64>, I<64>, I<64>, uint8_t>,
    std::tuple<I<64>, I<64>, I<64>, int32_t>,
    std::tuple<I<64>, I<64>, I<64>, uint32_t>,
    std::tuple<I<64>, I<64>, I<64>, bfloat16_t>,
    std::tuple<I<64>, I<64>, I<64>, hfloat16_t>,
    std::tuple<I<64>, I<64>, I<64>, float16_t>,
    std::tuple<I<64>, I<64>, I<64>, float32_t>,
    std::tuple<I<64>, I<64>, I<64>, float64_t>>;

TYPED_TEST_SUITE(FillFragmentTest, Implementations);

TYPED_TEST(FillFragmentTest, FillFragment)
{
    TypeParam types;
    test_fill_fragment(types);
};
