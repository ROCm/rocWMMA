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
#define PADDING_VALUE 999

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename LayoutA>
__global__ void loadContaminationTest(
    DataT const* a_in, DataT* a_out, uint32_t paddedM, uint32_t paddedN, uint32_t M, uint32_t N)
{
    using MappingA = MappingUtil<BlockM, BlockN, DataT, LayoutA>;

    int paddedlda = std::is_same<LayoutA, row_major>::value ? paddedN : paddedM;
    int lda       = std::is_same<LayoutA, row_major>::value ? N : M;

    // Create frags and fill
    auto fragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>();

    // Map, load and store.
    auto* readA  = MappingA::dataCoord(a_in, paddedlda);
    auto* writeA = MappingA::dataCoord(a_out, lda);
    wmma::load_matrix_sync(fragA, readA, paddedlda);
    wmma::store_matrix_sync(writeA, fragA, lda);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename LayoutA>
__global__ void storeContaminationTest(
    DataT const* a_in, DataT* a_out, uint32_t paddedM, uint32_t paddedN, uint32_t M, uint32_t N)
{
    using MappingA = MappingUtil<BlockM, BlockN, DataT, LayoutA>;

    int paddedlda = std::is_same<LayoutA, row_major>::value ? paddedN : paddedM;
    int lda       = std::is_same<LayoutA, row_major>::value ? N : M;

    // Create frags and fill
    auto fragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>();

    // Map, load and store.
    auto* readA  = MappingA::dataCoord(a_in, lda);
    auto* writeA = MappingA::dataCoord(a_out, paddedlda);
    wmma::load_matrix_sync(fragA, readA, lda);
    wmma::store_matrix_sync(writeA, fragA, paddedlda);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayout>
struct Kernel
{
public:
    Kernel(uint32_t TBlockXI, uint32_t TBlockYI, uint32_t MI, uint32_t NI)
    {
        TBlockX = TBlockXI;
        TBlockY = TBlockYI;
        M       = MI;
        N       = NI;

        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        std::cout << "HIP wmma::test_layout_util test: TBlock (" << TBlockX << ", " << TBlockY
                  << ") "
                  << "BlockMNK(" << BlockM << ", " << BlockN << "," << BlockK << ") "
                  << "MatrixMNK(" << M << ", " << N << ") "
                  << "FmtA(" << (std::is_same<DataLayout, row_major>::value ? "R" : "C") << ") "
                  << "TiTc(" << dataTypeToString<DataT>() << ") \n";

        gridDim
            = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));
        blockDim = dim3(TBlockX, TBlockY);
    }

    void LoadContaminationTestWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY)
            return;

        matrix.resize((M + 2) * (N + 2));
        MatrixUtil<DataLayout>::fill_with_padding(matrix, M + 2, N + 2, DataT(PADDING_VALUE));

        // Allocate and copy init values to device memory
        const size_t bytesP = (M + 2) * (N + 2) * sizeof(DataT);
        CHECK_HIP_ERROR(hipMalloc(&d_arr_in, bytesP));
        CHECK_HIP_ERROR(hipMemcpy(d_arr_in, matrix.data(), bytesP, hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipMalloc(&d_arr_out, (M) * (N) * sizeof(DataT)));

        hipLaunchKernelGGL((loadContaminationTest<BlockM, BlockN, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr_in + M + 3,
                           d_arr_out,
                           M + 2,
                           N + 2,
                           M,
                           N);

        //Initialize Ref matrix
        std::vector<DataT> ref((M * N), DataT(0));

        const size_t bytes = M * N * sizeof(DataT);
        CHECK_HIP_ERROR(hipMemcpy(ref.data(), d_arr_out, bytes, hipMemcpyDeviceToHost));

        //Compare
        EXPECT_TRUE(
            (compareEqualPadded<DataT, DataLayout>(ref, matrix, M, N, DataT(PADDING_VALUE))));
        CHECK_HIP_ERROR(hipFree(d_arr_in));
        CHECK_HIP_ERROR(hipFree(d_arr_out));
    }

    void StoreContaminationTestWrapper()
    {
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < TBlockY * BlockN)
            return;

        matrix.resize(M * N);
        MatrixUtil<DataLayout>::fill(matrix, M, N);

        // Allocate and copy init values to device memory
        const size_t bytes = (M) * (N) * sizeof(DataT);
        CHECK_HIP_ERROR(hipMalloc(&d_arr_in, bytes));
        CHECK_HIP_ERROR(hipMemcpy(d_arr_in, matrix.data(), bytes, hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipMalloc(&d_arr_out, (M + 2) * (N + 2) * sizeof(DataT)));
        std::vector<DataT> d_arr_out_padded((M + 2) * (N + 2), DataT(PADDING_VALUE));
        CHECK_HIP_ERROR(hipMemcpy(d_arr_out,
                                  d_arr_out_padded.data(),
                                  (M + 2) * (N + 2) * sizeof(DataT),
                                  hipMemcpyHostToDevice));

        hipLaunchKernelGGL((storeContaminationTest<BlockM, BlockN, BlockK, DataT, DataLayout>),
                           gridDim,
                           blockDim,
                           0, // sharedMemBytes
                           0, // stream
                           d_arr_in,
                           d_arr_out + M + 3,
                           M + 2,
                           N + 2,
                           M,
                           N);

        //Initialize Ref matrix
        std::vector<DataT> ref(((M + 2) * (N + 2)), DataT(0));
        const size_t       bytesP = (M + 2) * (N + 2) * sizeof(DataT);
        CHECK_HIP_ERROR(hipMemcpy(ref.data(), d_arr_out, bytesP, hipMemcpyDeviceToHost));

        //Compare
        EXPECT_TRUE(
            (compareEqualPadded<DataT, DataLayout>(matrix, ref, M, N, DataT(PADDING_VALUE))));
        CHECK_HIP_ERROR(hipFree(d_arr_in));
        CHECK_HIP_ERROR(hipFree(d_arr_out));
    }

    ~Kernel() {}

private:
    uint32_t           M, N, TBlockX, TBlockY;
    DataT *            d_arr_in, *d_arr_out;
    dim3               gridDim, blockDim;
    std::vector<DataT> matrix;
};

template <typename T>
struct ContaminationTestWrapper;

template <typename BlockM, typename BlockN, typename BlockK, typename DataT, typename DataLayout>
struct ContaminationTestWrapper<std::tuple<BlockM, BlockN, BlockK, DataT, DataLayout>>
    : public testing::Test
{
    Kernel<BlockM::value, BlockN::value, BlockK::value, DataT, DataLayout>* obj[NO_OF_TESTS];

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
                obj[i++]
                    = new Kernel<BlockM::value, BlockN::value, BlockK::value, DataT, DataLayout>(
                        tblock[0], tblock[1], size[0], size[1]);
            }
        }
    }

    void LoadContaminationSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->LoadContaminationTestWrapper();
    }

    void StoreContaminationSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->StoreContaminationTestWrapper();
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
    // BlockM, BlockN, BlockK, DataT, DataLayout/*
    std::tuple<I<16>, I<16>, I<16>, float32_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, float32_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, float16_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, float16_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, int8_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, int8_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, int32_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, int32_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, uint8_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, uint8_t, col_major>,
    std::tuple<I<16>, I<16>, I<16>, uint32_t, row_major>,
    std::tuple<I<16>, I<16>, I<16>, uint32_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, float16_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, float16_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, int8_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, int8_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, int32_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, int32_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, uint8_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, uint8_t, col_major>,
    std::tuple<I<32>, I<32>, I<32>, uint32_t, row_major>,
    std::tuple<I<32>, I<32>, I<32>, uint32_t, col_major>>;

TYPED_TEST_SUITE(ContaminationTestWrapper, Implementations);

TYPED_TEST(ContaminationTestWrapper, load)
{
    this->LoadContaminationSetup();
}

TYPED_TEST(ContaminationTestWrapper, store)
{
    this->StoreContaminationSetup();
}
