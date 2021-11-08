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

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
__global__ void col(DataT* out, uint32_t M, uint32_t N)
{
    enum : uint32_t
    {
        MaxVectorWidth    = VecWidthTraits<BlockDim, BlockK, DataT>::MaxVectorWidth,
        ElementsPerThread = std::is_same<DataLayout, row_major>::value ? MaxVectorWidth : 1
    };

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using LayoutT  = Layout::Col<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;
    using Mapping  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;

    int ldm = std::is_same<DataLayout, row_major>::value ? N : M;

    auto baseOffset  = LayoutT::baseDataOffset(ldm);
    auto iocount     = IOTraits::IOCount;
    auto matrixCoord = Mapping::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<DataLayout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<DataLayout, row_major>::value ? 1 : 0
    };

    for(uint32_t i = 0; i < iocount; ++i)
    {
        for(int j = 0; j < ElementsPerThread; j++)
        {
            auto index
                = (std::get<MajorIndex>(matrixCoord) * ldm + std::get<MinorIndex>(matrixCoord))
                  + baseOffset + j;
            out[index] = index;
        }
        baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
    }
}

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
__global__ void colNT(DataT* out, uint32_t M, uint32_t N)
{
    enum : uint32_t
    {
        MaxVectorWidth    = VecWidthTraits<BlockDim, BlockK, DataT>::MaxVectorWidth,
        ElementsPerThread = std::is_same<DataLayout, row_major>::value ? MaxVectorWidth : 1
    };

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using LayoutT
        = Layout::ColNT<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread, MaxVectorWidth>;
    using Mapping = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;

    int ldm = std::is_same<DataLayout, row_major>::value ? N : M;

    auto baseOffset  = LayoutT::baseDataOffset(ldm);
    auto iocount     = IOTraits::IOCount;
    auto matrixCoord = Mapping::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<DataLayout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<DataLayout, row_major>::value ? 1 : 0
    };

    for(uint32_t i = 0; i < iocount; ++i)
    {
        for(int j = 0; j < ElementsPerThread; j++)
        {
            auto index
                = (std::get<MajorIndex>(matrixCoord) * ldm + std::get<MinorIndex>(matrixCoord))
                  + baseOffset + j;
            out[index] = index;
        }
        baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
    }
}

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
__global__ void row(DataT* out, uint32_t M, uint32_t N)
{
    enum : uint32_t
    {
        MaxVectorWidth    = VecWidthTraits<BlockDim, BlockK, DataT>::MaxVectorWidth,
        ElementsPerThread = std::is_same<DataLayout, row_major>::value ? MaxVectorWidth : 1
    };

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using LayoutT  = Layout::Row<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;
    using Mapping  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;

    int ldm = std::is_same<DataLayout, row_major>::value ? N : M;

    auto baseOffset  = LayoutT::baseDataOffset(ldm);
    auto iocount     = IOTraits::IOCount;
    auto matrixCoord = Mapping::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<DataLayout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<DataLayout, row_major>::value ? 1 : 0
    };

    for(uint32_t i = 0; i < iocount; ++i)
    {
        for(int j = 0; j < ElementsPerThread; j++)
        {
            auto index
                = (std::get<MajorIndex>(matrixCoord) * ldm + std::get<MinorIndex>(matrixCoord))
                  + baseOffset + j;
            out[index] = index;
        }
        baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
    }
}

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
__global__ void rowNT(DataT* out, uint32_t M, uint32_t N)
{
    enum : uint32_t
    {
        MaxVectorWidth    = VecWidthTraits<BlockDim, BlockK, DataT>::MaxElementsPerThread,
        ElementsPerThread = std::is_same<DataLayout, row_major>::value ? MaxVectorWidth : 1
    };

    if((std::is_same<DataLayout, row_major>::value && ElementsPerThread > 1))
        return;

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using LayoutT
        = Layout::RowNT<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread, MaxVectorWidth>;
    using Mapping = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;

    int ldm = std::is_same<DataLayout, row_major>::value ? N : M;

    auto baseOffset  = LayoutT::baseDataOffset(ldm);
    auto iocount     = IOTraits::IOCount;
    auto matrixCoord = Mapping::matrixCoord();

    enum : uint32_t
    {
        MajorIndex = std::is_same<DataLayout, row_major>::value ? 0 : 1,
        MinorIndex = std::is_same<DataLayout, row_major>::value ? 1 : 0
    };

    for(uint32_t i = 0; i < iocount; ++i)
    {
        for(int j = 0; j < ElementsPerThread; j++)
        {
            auto index
                = (std::get<MajorIndex>(matrixCoord) * ldm + std::get<MinorIndex>(matrixCoord))
                  + baseOffset + j;
            out[index] = index;
        }
        baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
    }
}

template <typename T>
void GenerateLayoutIds(T data, int M, int N)
{
    for(uint32_t i = 0; i < M; i++)
        for(uint32_t j = 0; j < N; j++)
            data[i * M + j] = i * M + j;
}

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct Kernel
{
public:
    Kernel(uint32_t TBlockXI, uint32_t TBlockYI, uint32_t MI, uint32_t NI)
    {
        TBlockX = TBlockXI;
        TBlockY = TBlockYI;
        M       = MI;
        N       = NI;

        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        std::cout << "HIP wmma::test_layout_util test: TBlock (" << TBlockX << ", " << TBlockY
                  << ") "
                  << "BlockDimN(" << BlockDim << ", " << BlockK << ") "
                  << "MatrixMNK(" << M << ", " << N << ") "
                  << "FmtA(" << (std::is_same<DataLayout, row_major>::value ? "R" : "C") << ") "
                  << "TiTc(" << dataTypeToString<DataT>() << ") \n";

        // Generate Layout Matrix
        matrix.resize(M * N);
        GenerateLayoutIds(matrix.data(), M, N);

        // Allocate and copy init values to device memory
        const size_t bytes = M * N * sizeof(DataT);
        CHECK_HIP_ERROR(hipMalloc(&d_arr, bytes));
        CHECK_HIP_ERROR(hipMemset(d_arr, 0, bytes));

        gridDim
            = dim3(ceilDiv(M, BlockDim * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockK * TBlockY));
        blockDim = dim3(TBlockX, TBlockY);
    }

    void colWrapper()
    {
        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        hipLaunchKernelGGL((col<BlockDim, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr,
                           M,
                           N);
    }
    void colNTWrapper()
    {
        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        hipLaunchKernelGGL((colNT<BlockDim, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr,
                           M,
                           N);
    }

    void rowWrapper()
    {
        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        hipLaunchKernelGGL((row<BlockDim, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr,
                           M,
                           N);
    }

    void rowNTWrapper()
    {
        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        hipLaunchKernelGGL((rowNT<BlockDim, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr,
                           M,
                           N);
    }

    ~Kernel()
    {
        if(M < BlockDim * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockK)
            return;

        //Initialize Ref matrix
        std::vector<DataT> ref(M * N, 0);

        const size_t bytes = M * N * sizeof(DataT);
        CHECK_HIP_ERROR(hipMemcpy(ref.data(), d_arr, bytes, hipMemcpyDeviceToHost));

        // Compare
        auto compResult = compareEqual<DataT, DataT, DataLayout, DataLayout>(matrix, ref, M, N);
        EXPECT_TRUE((std::get<0>(compResult))) << std::get<1>(compResult);
        CHECK_HIP_ERROR(hipFree(d_arr));
    }

private:
    uint32_t           M, N, TBlockX, TBlockY;
    DataT*             d_arr;
    dim3               gridDim, blockDim;
    std::vector<DataT> matrix;
};

template <typename T>
struct LayoutWrapper;

template <typename BlockDim, typename BlockK, typename DataT, typename DataLayout>
struct LayoutWrapper<std::tuple<BlockDim, BlockK, DataT, DataLayout>> : public testing::Test
{
    Kernel<BlockDim::value, BlockK::value, DataT, DataLayout>* obj[NO_OF_TESTS];

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
                obj[i++] = new Kernel<BlockDim::value, BlockK::value, DataT, DataLayout>(
                    tblock[0], tblock[1], size[0], size[1]);
            }
        }
    }

    void colSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->colWrapper();
    }

    void colNTSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->colNTWrapper();
    }

    void rowSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->rowWrapper();
    }

    void rowNTSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->rowNTWrapper();
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
    // BlockDim, BlockK, DataT, DataLayout/*
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

TYPED_TEST_SUITE(LayoutWrapper, Implementations);

TYPED_TEST(LayoutWrapper, col)
{
    this->colSetup();
}

TYPED_TEST(LayoutWrapper, colNT)
{
    this->colNTSetup();
}

TYPED_TEST(LayoutWrapper, row)
{
    this->rowSetup();
}

#if 0
TYPED_TEST(LayoutWrapper, rowNT)
{
    this->rowNTSetup();
}
#endif
