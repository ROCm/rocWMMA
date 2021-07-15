
#include <hip/hip_runtime.h>

#include <type_traits>
#include <unistd.h>
#include <random>
#include <utility>
#include "Constants.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"
#include <gtest/gtest.h>

#include "Common.hpp"

enum functionType
{
    block2matrixT,
    wave2matrixT,
    matrix2dataT,
    thread2matrixT,
    matrix2dataOverrideMT,
    matrix2dataOverrideNT
};

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void block2matrix(InputT*   a_in,
                             InputT* a_out,
                             uint32_t  M,
                             uint32_t  N)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;
    typename MappingA::CoordT aCoord = MappingA::blockCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(col * BlockM + j) * lda + (row * BlockN + i)] =
                                a_in[(col * BlockM + j) * lda + (row * BlockN + i)];
}

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void wave2matrix(InputT*   a_in,
                            InputT* a_out,
                            uint32_t  M,
                            uint32_t  N)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;
    typename MappingA::CoordT aCoord = MappingA::waveCoord();
    typename MappingA::CoordT aCoord_wg = MappingA::WaveSpace::workgroupCoord();
    typename MappingA::CoordT aCoord_wgdim = MappingA::WaveSpace::workgroupDim();

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord) + (std::get<MajorIndex>(aCoord_wg)
                            * std::get<MajorIndex>(aCoord_wgdim));
    uint32_t row = std::get<MinorIndex>(aCoord) + (std::get<MinorIndex>(aCoord_wg)
                            * std::get<MinorIndex>(aCoord_wgdim));
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(col * BlockM + j) * lda + (row * BlockN + i)] =
                    a_in[(col * BlockM + j) * lda + (row * BlockN + i)];
}

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void matrix2data(InputT* a_in,
                              InputT* a_out,
                              uint32_t  M,
                              uint32_t  N)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;
    typename MappingA::CoordT aCoord = MappingA::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(col + j) * lda + (row + i)] =
                    a_in[(col + j) * lda + (row + i)];
}


template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void thread2matrix(InputT* a_in,
                              InputT* a_out,
                              uint32_t  M,
                              uint32_t  N)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    uint32_t minor = std::is_same<LayoutA, row_major>::value ?
                            (threadIdx.y + blockDim.y * blockIdx.y) :
                            ((threadIdx.x + blockDim.x * blockIdx.x)/AMDGCN_WAVE_SIZE);
    uint32_t major = std::is_same<LayoutA, row_major>::value ?
                            ((threadIdx.x + blockDim.x * blockIdx.x)/AMDGCN_WAVE_SIZE) :
                            (threadIdx.y + blockDim.y * blockIdx.y);

    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(minor * BlockM + j) + (major * BlockN + i) * lda ] =
                        a_in[(minor * BlockM + j) + (major * BlockN + i) * lda];
}

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void matrix2dataOverrideM(InputT* a_in,
                                     InputT* a_out,
                                     uint32_t  M,
                                     uint32_t  N,
                                     uint32_t mOverRide)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    typename MappingA::CoordT aCoord = MappingA::matrixCoordM(mOverRide);

    uint32_t col = std::is_same<LayoutA, row_major>::value ? std::get<MajorIndex>(aCoord)
                         : std::get<MinorIndex>(aCoord);
    uint32_t row = std::is_same<LayoutA, row_major>::value ? std::get<MinorIndex>(aCoord)
                         : std::get<MajorIndex>(aCoord);

    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(col + j) * lda + (row + i)] =
                    a_in[(col + j) * lda + (row + i)];
}

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__global__ void matrix2dataOverrideN(InputT* a_in,
                                     InputT* a_out,
                                     uint32_t  M,
                                     uint32_t  N,
                                     uint32_t nOverRide)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, LayoutA>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<LayoutA, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<LayoutA, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<LayoutA, row_major>::value ? N : M;

    typename MappingA::CoordT aCoord = MappingA::matrixCoordN(nOverRide);

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
                a_out[(col + j) * lda + (row + i)] =
                    a_in[(col + j) * lda + (row + i)];
}

template <typename T>
void GenerateTestData(T *data, int numElements)
{
    if(std::is_integral<T>::value)
    {
        for(size_t i = 0; i < numElements; i++)
        {
            data[i] = static_cast<T>(static_cast<int>(rand())/static_cast<int>(RAND_MAX/10000));
        }
    }
    else
    {
        for(size_t i = 0; i < numElements; i++)
        {
            data[i] = static_cast<T>(static_cast<float>(rand())/static_cast<float>(RAND_MAX/10000));
        }
    }
}

template <uint32_t BlockM,
          uint32_t BlockN,
          typename InputT,
          typename LayoutA>
__host__ void test_map_util_h(uint32_t TBlockX,
                                uint32_t TBlockY,
                                uint32_t M,
                                uint32_t N,
                                int functionType)
{
    if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN )
        return;

    std::cout << "HIP wmma::test_map_util test: TBlock (" << TBlockX << ", " << TBlockY << ") "
              << "BlockMN(" << BlockM << ", " << BlockN << ") "
              << "MatrixMNK(" << M << ", " << N << ") "
              << "FmtA(" << (std::is_same<LayoutA, row_major>::value ? "R" : "C") <<  ") "
              << "TiTc(" << dataTypeToString<InputT>()
              << ") \n";

    // Default Override testcase
    int mOverride = 32, nOverride = 32;

    // Initialize input matrix
    std::vector<InputT>   matrixA(M * N);
    srand(time(NULL));

    //Fill Matrices with random values
    GenerateTestData<InputT>(matrixA.data(), M * N);

    // Allocate and copy init values to device memory
    InputT      *d_a_in, *d_a_out;
    const size_t bytesA = matrixA.size() * sizeof(InputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a_in, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_a_out, bytesA));
    CHECK_HIP_ERROR(hipMemcpy(d_a_in, matrixA.data(), bytesA, hipMemcpyHostToDevice));

    auto gridDim
        = dim3(ceilDiv(M,  BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

    auto blockDim = dim3(TBlockX, TBlockY);

    std::cout << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;

    switch(functionType)
    {
        case block2matrixT:
        {
            hipLaunchKernelGGL(
            (block2matrix<BlockM, BlockN, InputT, LayoutA>),
            gridDim,
            blockDim,
            0, // sharedMemBytes
            0, // stream
            d_a_in,
            d_a_out,
            M,
            N);
            break;
        }

        case wave2matrixT:
        {
            hipLaunchKernelGGL(
            (wave2matrix<BlockM, BlockN, InputT, LayoutA>),
            gridDim,
            blockDim,
            0, // sharedMemBytes
            0, // stream
            d_a_in,
            d_a_out,
            M,
            N);
            break;
        }

        case matrix2dataT:
        {
            hipLaunchKernelGGL(
            (matrix2data<BlockM, BlockN, InputT, LayoutA>),
            gridDim,
            blockDim,
            0, // sharedMemBytes
            0, // stream
            d_a_in,
            d_a_out,
            M,
            N);
            break;
        }

        case thread2matrixT:
        {
            hipLaunchKernelGGL(
            (thread2matrix<BlockM, BlockN, InputT, LayoutA>),
            gridDim,
            blockDim,
            0, // sharedMemBytes
            0, // stream
            d_a_in,
            d_a_out,
            M,
            N);
            break;
        }

        default:
            break;
    }

    //Initialize Ref matrix
    std::vector<InputT>   refA(M * N, 0);

    CHECK_HIP_ERROR(hipMemcpy(refA.data(), d_a_out, bytesA, hipMemcpyDeviceToHost));

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a_in));
    CHECK_HIP_ERROR(hipFree(d_a_out));

    //Compare
    EXPECT_TRUE((compareEqual<InputT, InputT, LayoutA, LayoutA>(matrixA, refA, M, N)));
}

template <typename IntConstBlockM,
          typename IntConstBlockN,
          typename InputT>
__host__ void test_map_util_h(uint32_t TBlockX,
                                uint32_t TBlockY,
                                uint32_t M,
                                uint32_t N,
                                int functionType)
{
    std::tuple<row_major, col_major> types;
    for_each(types, [&](auto layout_a) {
                test_map_util_h<IntConstBlockM::value,
                                IntConstBlockN::value,
                                InputT,
                                decltype(layout_a)>(
                    TBlockX, TBlockY, M, N, functionType);
    });
}

template <typename... Ts>
void test_map_util(std::tuple<Ts...>, int functionType)
{
    // clang-format off
    std::vector<std::array<int, 2>> thread_block = {{64,1}, {64, 2}, {64, 4}, {64, 8}, {64, 16},
                                                    {128,1}, {128,2}, {128,4}, {128,8},
                                                    {256,1}, {256,2}, {256,4},
                                                    {512,1}, {512,2}};

    // For fills, we must have the same geometry for all matrices
    std::vector<std::array<int, 2>> problem_sizes = {{16, 16},
                                                     {32, 32},
                                                     {64, 64},
                                                     {256, 256},
                                                     {512, 512},
                                                     {1024, 1024},
                                                     {2048, 2048}};
    // clang-format on
    for(auto tblock : thread_block)
    {
        for(auto size : problem_sizes)
        {
            auto fargs = std::tuple_cat(tblock, size, std::tie(functionType));
            std::apply(test_map_util_h<Ts...>, fargs);
        }
    }
}

template <typename T>
struct MappingUtilTest : public testing::Test
{
};

using Implementations = testing::Types<
    // BlockM, BlockN, InputT
    std::tuple<I<16>, I<16>, float32_t>,
    std::tuple<I<16>, I<16>, float16_t>,
    std::tuple<I<16>, I<16>, hfloat16_t>,
    std::tuple<I<16>, I<16>, int8_t>,
    std::tuple<I<16>, I<16>, int32_t>,
    std::tuple<I<16>, I<16>, uint8_t>,
    std::tuple<I<16>, I<16>, uint32_t>,
    std::tuple<I<32>, I<32>, float32_t>,
    std::tuple<I<32>, I<32>, float16_t>,
    std::tuple<I<32>, I<32>, hfloat16_t>,
    std::tuple<I<32>, I<32>, int8_t>,
    std::tuple<I<32>, I<32>, int32_t>,
    std::tuple<I<32>, I<32>, uint8_t>,
    std::tuple<I<32>, I<32>, uint32_t>>;

TYPED_TEST_SUITE(MappingUtilTest, Implementations);

TYPED_TEST(MappingUtilTest, Block2Matrix)
{
    TypeParam types;
    test_map_util(types, block2matrixT);
};

TYPED_TEST(MappingUtilTest, Wave2Matrix)
{
    TypeParam types;
    test_map_util(types, wave2matrixT);
};

TYPED_TEST(MappingUtilTest, Matrix2Data)
{
    TypeParam types;
    test_map_util(types, matrix2dataT);
};

TYPED_TEST(MappingUtilTest, Thread2Matrix)
{
    TypeParam types;
    test_map_util(types, thread2matrixT);
};
