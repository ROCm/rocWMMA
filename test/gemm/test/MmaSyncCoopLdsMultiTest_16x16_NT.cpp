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

#include "MmaSyncCoopLdsMultiTest.h"

// Test params for 16 x 16 NT kernels
struct TestParams16x16NT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::col_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL + double
    // Block Sizes: 16 x 16 x BlockK
    // Layouts: NT
    using Types      = std::tuple<std::tuple<float16_t, float32_t, float32_t>>;
    using BlockSizes = typename Base::TestBlockSizes16x16;
    using Layouts    = typename CombineOne<ABLayouts, typename Base::TestLayoutTypes>::Result;
    using BlocksXY   = std::tuple<std::tuple<I<2>, I<2>>>;

    // Assemble the kernel generator
    using TestParams = typename CombineMany<
        Types,
        typename CombineMany<BlockSizes,
                             typename CombineMany<Layouts, BlocksXY>::Result>::Result>::Result;
    using GeneratorImpl   = MmaSyncCoopLdsMultiGenerator;
    using KernelGenerator = KernelGenerator<TestParams, GeneratorImpl>;

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class MmaSyncCoopLdsMultiTest16x16NT : public MmaSyncCoopLdsMultiTest
{
};

TEST_P(MmaSyncCoopLdsMultiTest16x16NT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncCoopLdsMultiTest16x16NT,
                         ::testing::Combine(::testing::ValuesIn(TestParams16x16NT::kernels()),
                                            ::testing::ValuesIn(TestParams16x16NT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams16x16NT::problemSizes()),
                                            ::testing::ValuesIn(TestParams16x16NT::alphas()),
                                            ::testing::ValuesIn(TestParams16x16NT::betas())));
