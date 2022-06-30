/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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

#include <type_traits>

#include "detail/mma_sync_coop_wg.hpp"
#include "gemm_config.hpp"
#include "gemm_test.hpp"
#include "kernel_generator.hpp"

namespace rocwmma
{

    struct TestParams : public CommonTestParams
    {
        using Base = CommonTestParams;

        // Types: Small types
        // Block Sizes: 32 x 32 x BlockK
        // Layouts: TT
        using Types       = typename Base::TestTypesSmall;
        using BlockSizes  = std::tuple<std::tuple<I<32>, I<32>, I<8>>>;
        using Layouts     = typename Base::TestLayoutsTT;
        using LayoutsLds  = typename Base::TestLdsLayoutTypes;
        using GemmConfigs = typename Base::TestGemmConfigsWgLevel;
        using BlocksXY    = std::tuple<std::tuple<I<4>, I<4>>>;
        using KernelParams =
            typename CombineLists<Types, BlockSizes, Layouts, LayoutsLds, GemmConfigs, BlocksXY>::
                Result;

        // Assemble the kernel generator
        // Kernel: MmaSyncMulti
        using GeneratorImpl   = MmaSyncCoopWgGenerator;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

} // namespace rocwmma

// Test suite for unique parameterization
class MmaSyncCoopWgTest32x32TT4x4 : public rocwmma::GemmTest
{
};

TEST_P(MmaSyncCoopWgTest32x32TT4x4, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    GemmKernelTests,
    MmaSyncCoopWgTest32x32TT4x4,
    ::testing::Combine(::testing::ValuesIn(rocwmma::TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::TestParams::alphas()),
                       ::testing::ValuesIn(rocwmma::TestParams::betas())));
