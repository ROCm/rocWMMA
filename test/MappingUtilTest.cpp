/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hip/hip_runtime.h>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"
#include <random>
#include <type_traits>
#include <unistd.h>
#include <utility>

#include "WMMA.h"
#include <gtest/gtest.h>

#include "Common.hpp"
#define NO_OF_TESTS 98
#define INVALID_VALUE -123

template <typename T>
void GenerateTestData(T* data, int numElements)
{
    if(std::is_integral<T>::value)
    {
        for(size_t i = 0; i < numElements; i++)
        {
            data[i] = static_cast<T>(static_cast<int>(rand()) / static_cast<int>(RAND_MAX / 10000));
        }
    }
    else
    {
        for(size_t i = 0; i < numElements; i++)
        {
            data[i]
                = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 10000));
        }
    }
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void block2matrix(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N)
{
    using MappingA                   = MappingUtil<BlockM, BlockN, InputT, Layout>;
    typename MappingA::CoordT aCoord = MappingA::blockCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(col * BlockM + j) * lda + (row * BlockN + i)]
                = a_in[(col * BlockM + j) * lda + (row * BlockN + i)];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void wave2matrix(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N)
{
    using MappingA                         = MappingUtil<BlockM, BlockN, InputT, Layout>;
    typename MappingA::CoordT aCoord       = MappingA::waveCoord();
    typename MappingA::CoordT aCoord_wg    = MappingA::WaveSpace::workgroupCoord();
    typename MappingA::CoordT aCoord_wgdim = MappingA::WaveSpace::workgroupDim();

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord)
                   + (std::get<MajorIndex>(aCoord_wg) * std::get<MajorIndex>(aCoord_wgdim));
    uint32_t row = std::get<MinorIndex>(aCoord)
                   + (std::get<MinorIndex>(aCoord_wg) * std::get<MinorIndex>(aCoord_wgdim));
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(col * BlockM + j) * lda + (row * BlockN + i)]
                = a_in[(col * BlockM + j) * lda + (row * BlockN + i)];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void matrix2data(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N)
{
    using MappingA                   = MappingUtil<BlockM, BlockN, InputT, Layout>;
    typename MappingA::CoordT aCoord = MappingA::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(col + j) * lda + (row + i)] = a_in[(col + j) * lda + (row + i)];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void thread2matrix(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, Layout>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    uint32_t minor = std::is_same<Layout, row_major>::value
                         ? (threadIdx.y + blockDim.y * blockIdx.y)
                         : ((threadIdx.x + blockDim.x * blockIdx.x) / AMDGCN_WAVE_SIZE);
    uint32_t major = std::is_same<Layout, row_major>::value
                         ? ((threadIdx.x + blockDim.x * blockIdx.x) / AMDGCN_WAVE_SIZE)
                         : (threadIdx.y + blockDim.y * blockIdx.y);

    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(minor * BlockM + j) + (major * BlockN + i) * lda]
                = a_in[(minor * BlockM + j) + (major * BlockN + i) * lda];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void
    matrix2dataOverrideM(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N, uint32_t mOverRide)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, Layout>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    typename MappingA::CoordT aCoord = MappingA::matrixCoordM(mOverRide);

    uint32_t col = std::is_same<Layout, row_major>::value ? std::get<MajorIndex>(aCoord)
                                                          : std::get<MinorIndex>(aCoord);
    uint32_t row = std::is_same<Layout, row_major>::value ? std::get<MinorIndex>(aCoord)
                                                          : std::get<MajorIndex>(aCoord);

    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(col + j) * lda + (row + i)] = a_in[(col + j) * lda + (row + i)];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
__global__ void
    matrix2dataOverrideN(InputT* a_in, InputT* a_out, uint32_t M, uint32_t N, uint32_t nOverRide)
{
    using MappingA = MappingUtil<BlockM, BlockN, InputT, Layout>;

    enum : uint32_t
    {
        MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0
    };

    int lda = std::is_same<Layout, row_major>::value ? N : M;

    typename MappingA::CoordT aCoord = MappingA::matrixCoordN(nOverRide);

    uint32_t col = std::get<MajorIndex>(aCoord);
    uint32_t row = std::get<MinorIndex>(aCoord);
    for(int i = 0; i < BlockM; i++)
        for(int j = 0; j < BlockN; j++)
            a_out[(col + j) * lda + (row + i)] = a_in[(col + j) * lda + (row + i)];
}

template <uint32_t BlockM, uint32_t BlockN, typename InputT, typename Layout>
struct Kernel
{
public:
    Kernel(uint32_t TBlockXI, uint32_t TBlockYI, uint32_t MI, uint32_t NI)
    {

        M       = MI;
        N       = NI;
        TBlockX = TBlockXI;
        TBlockY = TBlockYI;

        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || NI < TBlockY * BlockN)
            return;

        std::cout << "HIP wmma::test_map_util test: TBlock (" << TBlockX << ", " << TBlockY << ") "
                  << "BlockMN(" << BlockM << ", " << BlockN << ") "
                  << "MatrixMNK(" << M << ", " << NI << ") "
                  << "FmtA(" << (std::is_same<Layout, row_major>::value ? "R" : "C") << ") "
                  << "TiTc(" << dataTypeToString<InputT>() << ") \n";

        // Initialize input matrix
        matrixA.resize(M * N);
        srand(time(NULL));

        //Fill Matrices with random values
        GenerateTestData<InputT>(matrixA.data(), M * N);

        // Allocate and copy init values to device memory
        const size_t bytesA = matrixA.size() * sizeof(InputT);

        CHECK_HIP_ERROR(hipMalloc(&d_a_in, bytesA));
        CHECK_HIP_ERROR(hipMalloc(&d_a_out, bytesA));
        CHECK_HIP_ERROR(hipMemcpy(d_a_in, matrixA.data(), bytesA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(d_a_out, 0, bytesA));

        gridDim
            = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

        blockDim = dim3(TBlockX, TBlockY);
    }

    void block2matrixWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        hipLaunchKernelGGL((block2matrix<BlockM, BlockN, InputT, Layout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_a_in,
                           d_a_out,
                           M,
                           N);
    }

    void wave2matrixWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        hipLaunchKernelGGL((wave2matrix<BlockM, BlockN, InputT, Layout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_a_in,
                           d_a_out,
                           M,
                           N);
    }

    void matrix2dataWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        hipLaunchKernelGGL((matrix2data<BlockM, BlockN, InputT, Layout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_a_in,
                           d_a_out,
                           M,
                           N);
    }

    void thread2matrixWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        hipLaunchKernelGGL((thread2matrix<BlockM, BlockN, InputT, Layout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_a_in,
                           d_a_out,
                           M,
                           N);
    }

    ~Kernel()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        //Initialize Ref matrix
        std::vector<InputT> refA(M * N, INVALID_VALUE);

        const size_t bytesA = refA.size() * sizeof(InputT);

        CHECK_HIP_ERROR(hipMemcpy(refA.data(), d_a_out, bytesA, hipMemcpyDeviceToHost));

        // Release device memory
        CHECK_HIP_ERROR(hipFree(d_a_in));
        CHECK_HIP_ERROR(hipFree(d_a_out));

        //Compare
        auto compResult = compareEqual<InputT, InputT, Layout, Layout>(matrixA, refA, M, N);
        EXPECT_TRUE((std::get<0>(compResult))) << std::get<1>(compResult);
    }

private:
    uint32_t            M, N, TBlockX, TBlockY;
    InputT *            d_a_in, *d_a_out;
    dim3                gridDim, blockDim;
    std::vector<InputT> matrixA;
};

template <typename T>
struct SetupWrapper;

template <typename BlockM, typename BlockN, typename InputT, typename Layout>
struct SetupWrapper<std::tuple<BlockM, BlockN, InputT, Layout>> : public testing::Test
{
    Kernel<BlockM::value, BlockN::value, InputT, Layout>* obj[NO_OF_TESTS];

    void SetUp() override
    {
        std::vector<std::array<int, 2>> thread_block = {{64, 1},
                                                        {64, 2},
                                                        {64, 4},
                                                        {64, 8},
                                                        {64, 16},
                                                        {128, 1},
                                                        {128, 2},
                                                        {128, 4},
                                                        {128, 8},
                                                        {256, 1},
                                                        {256, 2},
                                                        {256, 4},
                                                        {512, 1},
                                                        {512, 2}};

        // For fills, we must have the same geometry for all matrices
        std::vector<std::array<int, 2>> problem_sizes
            = {{16, 16}, {32, 32}, {64, 64}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}};

        // clang-format on
        int i = 0;
        for(auto tblock : thread_block)
        {
            for(auto size : problem_sizes)
            {
                obj[i++] = new Kernel<BlockM::value, BlockN::value, InputT, Layout>(
                    tblock[0], tblock[1], size[0], size[1]);
            }
        }
    }

    void block2matrixWrapperSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->block2matrixWrapper();
    }

    void wave2matrixWrapperSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->wave2matrixWrapper();
    }

    void matrix2dataWrapperSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->matrix2dataWrapper();
    }

    void thread2matrixWrapperSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->thread2matrixWrapper();
    }

    void TearDown() override
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
        {
            if(obj[i] != NULL)
            {
                delete obj[i];
                obj[i] = NULL;
            }
        }
    }
};

using Implementations = testing::Types<
    // BlockM, BlockN, InputT, layout/*
    std::tuple<I<16>, I<16>, float32_t, row_major>,
    std::tuple<I<16>, I<16>, float32_t, col_major>,
    std::tuple<I<16>, I<16>, float16_t, row_major>,
    std::tuple<I<16>, I<16>, float16_t, col_major>,
    std::tuple<I<16>, I<16>, hfloat16_t, row_major>,
    std::tuple<I<16>, I<16>, hfloat16_t, col_major>,
    std::tuple<I<16>, I<16>, int8_t, row_major>,
    std::tuple<I<16>, I<16>, int8_t, col_major>,
    std::tuple<I<16>, I<16>, int32_t, row_major>,
    std::tuple<I<16>, I<16>, int32_t, col_major>,
    std::tuple<I<16>, I<16>, uint8_t, row_major>,
    std::tuple<I<16>, I<16>, uint8_t, col_major>,
    std::tuple<I<16>, I<16>, uint32_t, row_major>,
    std::tuple<I<16>, I<16>, uint32_t, col_major>,
    std::tuple<I<32>, I<32>, float32_t, row_major>,
    std::tuple<I<32>, I<32>, float32_t, col_major>,
    std::tuple<I<32>, I<32>, float16_t, row_major>,
    std::tuple<I<32>, I<32>, float16_t, col_major>,
    std::tuple<I<32>, I<32>, hfloat16_t, row_major>,
    std::tuple<I<32>, I<32>, hfloat16_t, col_major>,
    std::tuple<I<32>, I<32>, int8_t, row_major>,
    std::tuple<I<32>, I<32>, int8_t, col_major>,
    std::tuple<I<32>, I<32>, int32_t, row_major>,
    std::tuple<I<32>, I<32>, int32_t, col_major>,
    std::tuple<I<32>, I<32>, uint8_t, row_major>,
    std::tuple<I<32>, I<32>, uint8_t, col_major>,
    std::tuple<I<32>, I<32>, uint32_t, row_major>,
    std::tuple<I<32>, I<32>, uint32_t, col_major>>;

TYPED_TEST_SUITE(SetupWrapper, Implementations);

TYPED_TEST(SetupWrapper, block2matrix)
{
    this->block2matrixWrapperSetup();
}

TYPED_TEST(SetupWrapper, wave2matrix)
{
    this->wave2matrixWrapperSetup();
}

TYPED_TEST(SetupWrapper, matrix2data)
{
    this->matrix2dataWrapperSetup();
}

TYPED_TEST(SetupWrapper, thread2matrix)
{
    this->thread2matrixWrapperSetup();
}
