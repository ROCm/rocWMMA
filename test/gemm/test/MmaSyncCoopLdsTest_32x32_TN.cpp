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

#include "MmaSyncCoopLdsTest.h"

// Test params for 32 x 32 TN kernels
struct TestParams32x32TN : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::row_major, wmma::col_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL - double
    // Block Sizes: 32 x 32 x BlockK
    // Layouts: TN
    using Types      = typename Base::TestTypesIOC;
    using BlockSizes = typename Base::TestBlockSizes32x32;
    using Layouts    = typename CombineOne<ABLayouts, typename Base::TestLayoutTypes>::Result;

    // Assemble the kernel generator
    using TestParams =
        typename CombineMany<Types, typename CombineMany<BlockSizes, Layouts>::Result>::Result;
    using GeneratorImpl   = MmaSyncCoopLdsGenerator;
    using KernelGenerator = KernelGenerator<TestParams, GeneratorImpl>;

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class MmaSyncCoopLdsTest32x32TN : public MmaSyncCoopLdsTest
{
};

TEST_P(MmaSyncCoopLdsTest32x32TN, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncCoopLdsTest32x32TN,
                         ::testing::Combine(::testing::ValuesIn(TestParams32x32TN::kernels()),
                                            ::testing::ValuesIn(TestParams32x32TN::threadBlocks()),
                                            ::testing::ValuesIn(TestParams32x32TN::problemSizes()),
                                            ::testing::ValuesIn(TestParams32x32TN::alphas()),
                                            ::testing::ValuesIn(TestParams32x32TN::betas())));
