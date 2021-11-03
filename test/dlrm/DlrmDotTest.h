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

#include "CommonTestParams.h"
#include "DlrmKernelBase.h"
#include "KernelGenerator.h"

// Wrapper into the actual device function
template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize typename DataT>
struct DlrmDotKernel final : public DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>
{
private:
    using Base = DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>;

public:
    DlrmDotKernel() {}
    ~DlrmDotKernel() final {}

    typename Base::KernelFwdFunc kernelFwdImpl() const final
    {
        return typename Base::KernelFwdFunc(dotBasedInteractFwdKernel<warps_per_threadblock,
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
            typename Base::KernelFwdFunc(dotBasedInteractFwdKernelNonAligned<warps_per_threadblock,
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
        return
            typename Base::KernelBwdFunc(dotBasedInteractBwdKernel<kWarpsPerBlock,
                                                                   kNumThreads,
                                                                   kRowTilesPerStep,
                                                                   kColTilesPerStep,
                                                                   kWarpSize,
                                                                   kWarpSizeLog2,
                                                                   (TileSize ? TileSize : kTileDim),
                                                                   kTileDimLog2>);
    }

    typename Base::KernelBwdFunc kernelBwdNonAlignedImpl() const final
    {
        return typename Base::KernelBwdFunc(
            dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock,
                                                kNumThreads,
                                                kRowTilesPerStep,
                                                kColTilesPerStep,
                                                kWarpSize,
                                                kWarpSizeLog2,
                                                (TileSize ? TileSize : kTileDim),
                                                kTileDimLog2>);
    }
};

// This is the GeneratorImpl class
struct DlrmDotGenerator
{
    // Indices to test parameters
    enum : uint32_t
    {
        BlockM   = 0,
        BlockN   = 1,
        BlockK   = 2,
        TileSize = 3,
        DataT    = 4
    };

    using ResultT = std::shared_ptr<KernelI>;

    template <typename... Ts>
    static ResultT generate(std::tuple<Ts...> testParams)
    {
        // Map GTest params to Kernel params
        using TestParamsT = std::tuple<Ts...>;
        using KernelT
            = DlrmDotKernel<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                            std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                            std::tuple_element_t<BlockK, TestParamsT>::value, // BlockK
                            std::tuple_element_t<TileSize, TestParamsT>::value, // TileSize
                            std::tuple_element_t<DataT, TestParamsT>, // DataT
                            >;

        return std::make_shared<KernelT>();
    }
};

// Needs changes
struct DlrmDotTest
    : public ::testing::TestWithParam<std::tuple<typename DlrmDotGenerator::ResultT,
                                                 typename CommonTestParams::ThreadBlockT,
                                                 typename CommonTestParams::ProblemSizeT,
                                                 typename CommonTestParams::AlphaT,
                                                 typename CommonTestParams::BetaT>>
{
    using Base = ::testing::TestWithParam<std::tuple<typename MmaSyncGenerator::ResultT,
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
// pass enum template values through Base::<name>

#endif // DLRM_DOT_TEST_H
