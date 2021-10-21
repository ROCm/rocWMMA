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

#ifndef WMMA_GEMM_MMA_SYNC_COOP_LDS_TEST_H
#define WMMA_GEMM_MMA_SYNC_COOP_LDS_TEST_H

#include <gtest/gtest.h>

#include "device/MmaSyncCoopLds.h"

#include "CommonTestParams.h"
#include "GemmKernelBase.h"
#include "KernelGenerator.h"

// Wrapper into the actual device function
template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
struct MmaSyncCoopLdsKernel final : public GemmKernelBase<BlockM,
                                                          BlockN,
                                                          BlockK,
                                                          InputT,
                                                          OutputT,
                                                          ComputeT,
                                                          LayoutA,
                                                          LayoutB,
                                                          LayoutC,
                                                          LayoutD>
{
private:
    using Base = GemmKernelBase<BlockM,
                                BlockN,
                                BlockK,
                                InputT,
                                OutputT,
                                ComputeT,
                                LayoutA,
                                LayoutB,
                                LayoutC,
                                LayoutD>;

public:
    MmaSyncCoopLdsKernel() {}
    ~MmaSyncCoopLdsKernel() final {}

    // Lds memory usage in bytes
    uint32_t ldsUsage() const final
    {
        auto blockDims = this->blockDim();
        return sizeof(InputT)
               * (blockDims.x / AMDGCN_WAVE_SIZE * BlockM * BlockK + blockDims.y * BlockK * BlockN);
    }

    typename Base::KernelFunc kernelImpl() const final
    {
        return typename Base::KernelFunc(mmaSyncCoopLds<BlockM,
                                                        BlockN,
                                                        BlockK,
                                                        InputT,
                                                        OutputT,
                                                        ComputeT,
                                                        LayoutA,
                                                        LayoutB,
                                                        LayoutC,
                                                        LayoutD>);
    }
};

// This is the GeneratorImpl class for MmaSyncCoopLds
struct MmaSyncCoopLdsGenerator
{
    // Indices to test parameters
    enum : uint32_t
    {
        InputT   = 0,
        OutputT  = 1,
        ComputeT = 2,
        BlockM   = 3,
        BlockN   = 4,
        BlockK   = 5,
        LayoutA  = 6,
        LayoutB  = 7,
        LayoutCD = 8
    };

    using ResultT = std::shared_ptr<KernelI>;

    template <typename... Ts>
    static ResultT generate(std::tuple<Ts...> testParams)
    {
        using TestParamsT = std::tuple<Ts...>;
        using KernelT
            = MmaSyncCoopLdsKernel<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                   std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                   std::tuple_element_t<BlockK, TestParamsT>::value, // BlockK
                                   std::tuple_element_t<InputT, TestParamsT>, // InputT
                                   std::tuple_element_t<OutputT, TestParamsT>, // OutputT
                                   std::tuple_element_t<ComputeT, TestParamsT>, // ComputeT
                                   std::tuple_element_t<LayoutA, TestParamsT>, // LayoutA
                                   std::tuple_element_t<LayoutB, TestParamsT>, // LayoutB
                                   std::tuple_element_t<LayoutCD, TestParamsT>, // LayoutC
                                   std::tuple_element_t<LayoutCD, TestParamsT> // LayoutD
                                   >;

        return std::make_shared<KernelT>();
    }
};

struct MmaSyncCoopLdsTest
    : public ::testing::TestWithParam<std::tuple<typename MmaSyncCoopLdsGenerator::ResultT,
                                                 typename CommonTestParams::ThreadBlockT,
                                                 typename CommonTestParams::ProblemSizeT,
                                                 typename CommonTestParams::AlphaT,
                                                 typename CommonTestParams::BetaT>>
{
    using Base = ::testing::TestWithParam<std::tuple<typename MmaSyncCoopLdsGenerator::ResultT,
                                                     typename CommonTestParams::ThreadBlockT,
                                                     typename CommonTestParams::ProblemSizeT,
                                                     typename CommonTestParams::AlphaT,
                                                     typename CommonTestParams::BetaT>>;

    void        SetUp() final {}
    void        TearDown() final {}
    static void RunKernel()
    {
        // Construct ProblemParams from
        // incoming gtest parameterization
        auto param       = Base::GetParam();
        auto kernel      = std::get<0>(param);
        auto threadBlock = std::get<1>(param);
        auto problemSize = std::get<2>(param);
        auto alpha       = std::get<3>(param);
        auto beta        = std::get<4>(param);

        ProblemParams params = {threadBlock, problemSize, alpha, beta};

        // Walk through kernel workflow
        kernel->setup(params);
        kernel->exec();
        kernel->validateResults();
        kernel->reportResults();
        kernel->tearDown();
    }
};

#endif // WMMA_GEMM_MMA_SYNC_COOP_LDS_TEST_H
