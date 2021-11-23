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

#ifndef DLRM_DOT_TEST_H
#define DLRM_DOT_TEST_H

#include <gtest/gtest.h>

#include "DlrmDot.h"

#include "DlrmKernelBase.h"
#include "DlrmTestParams.h"
#include "KernelGenerator.h"

// Wrapper into the actual device function
template <uint32_t TileSize, typename DataT>
struct DlrmDotKernel final : public DlrmKernelBase<TileSize, DataT>
{
private:
    using Base = DlrmKernelBase<TileSize, DataT>;

public:
    DlrmDotKernel() {}
    ~DlrmDotKernel() final {}

    typename Base::KernelFwdFunc kernelFwdImpl() const final
    {
        return typename Base::KernelFwdFunc(dotBasedInteractFwdKernel<DataT,
                                                                      warps_per_threadblock,
                                                                      threadblock_size,
                                                                      M_BLOCKS,
                                                                      K_BLOCKS,
                                                                      SMEM_STRIDE,
                                                                      SMEM_STRIDE_ACC,
                                                                      kWarpSize,
                                                                      kWarpSizeLog2,
                                                                      kTileDim,
                                                                      kTileDimLog2>);
    }

    typename Base::KernelFwdFunc kernelFwdNonAlignedImpl() const final
    {
        return
            typename Base::KernelFwdFunc(dotBasedInteractFwdKernelNonAligned<DataT,
                                                                             warps_per_threadblock,
                                                                             threadblock_size,
                                                                             M_BLOCKS,
                                                                             K_BLOCKS,
                                                                             SMEM_STRIDE,
                                                                             SMEM_STRIDE_ACC,
                                                                             kWarpSize,
                                                                             kWarpSizeLog2,
                                                                             kTileDim,
                                                                             kTileDimLog2>);
    }

    typename Base::KernelBwdFunc kernelBwdImpl() const final
    {
        return typename Base::KernelBwdFunc(dotBasedInteractBwdKernel<DataT,
                                                                      kWarpsPerBlock,
                                                                      kNumThreads,
                                                                      32 / TileSize,
                                                                      kColTilesPerStep,
                                                                      kWarpSize,
                                                                      kWarpSizeLog2,
                                                                      TileSize,
                                                                      Log2<TileSize>::value>);
    }

    typename Base::KernelBwdFunc kernelBwdNonAlignedImpl() const final
    {
        return typename Base::KernelBwdFunc(
            dotBasedInteractBwdKernelNonAligned<DataT,
                                                kWarpsPerBlock,
                                                kNumThreads,
                                                32 / TileSize,
                                                kColTilesPerStep,
                                                kWarpSize,
                                                kWarpSizeLog2,
                                                TileSize,
                                                Log2<TileSize>::value>);
    }
};

// This is the GeneratorImpl class
struct DlrmDotGenerator
{
    // Indices to test parameters
    enum : uint32_t
    {
        DataT    = 0,
        TileSize = 1
    };

    using ResultT = std::shared_ptr<KernelI>;

    template <typename... Ts>
    static ResultT generate(std::tuple<Ts...> testParams)
    {
        // Map GTest params to Kernel params
        using TestParamsT = std::tuple<Ts...>;
        using KernelT
            = DlrmDotKernel<std::tuple_element_t<TileSize, TestParamsT>::value, // TileSize
                            std::tuple_element_t<DataT, TestParamsT> // DataT
                            >;

        return std::make_shared<KernelT>();
    }
};

struct DlrmDotTest
    : public ::testing::TestWithParam<std::tuple<typename DlrmTestParams::KernelT,
                                                 typename DlrmTestParams::ThreadBlockT,
                                                 typename DlrmTestParams::ProblemSizeT,
                                                 typename DlrmTestParams::FwdDataSizeT,
                                                 typename DlrmTestParams::BwdDataSizeT,
                                                 typename DlrmTestParams::PassDirectionT>>
{
    using Base = ::testing::TestWithParam<std::tuple<typename DlrmTestParams::KernelT,
                                                     typename DlrmTestParams::ThreadBlockT,
                                                     typename DlrmTestParams::ProblemSizeT,
                                                     typename DlrmTestParams::FwdDataSizeT,
                                                     typename DlrmTestParams::BwdDataSizeT,
                                                     typename DlrmTestParams::PassDirectionT>>;

    void SetUp() override
    {
        // Construct ProblemParams from
        // incoming gtest parameterization
        auto param       = Base::GetParam();
        auto kernel      = std::get<0>(param);
        auto threadBlock = std::get<1>(param);
        auto problemSize = std::get<2>(param);
        auto fwdDataSize = std::get<3>(param);
        auto bwdDataSize = std::get<4>(param);
        auto isBwd       = std::get<5>(param);

        ProblemParams params = {threadBlock, problemSize, fwdDataSize, bwdDataSize, isBwd};

        // Walk through kernel workflow
        kernel->setup(params);

        // run warm-up iteration based on static bool (runkernel)
    }

    virtual void RunKernel()
    {
        // Construct ProblemParams from
        // incoming gtest parameterization
        auto param  = Base::GetParam();
        auto kernel = std::get<0>(param);
        kernel->exec();
        kernel->validateResults();
        kernel->reportResults();
    }

    virtual void Warmup()
    {
        auto param  = Base::GetParam();
        auto kernel = std::get<0>(param);
        kernel->exec();
    }

    void TearDown() override
    {
        // Construct ProblemParams from
        // incoming gtest parameterization
        auto param  = Base::GetParam();
        auto kernel = std::get<0>(param);
        kernel->tearDown();
    }
};
// pass enum template values through Base::<name>

#endif // DLRM_DOT_TEST_H
