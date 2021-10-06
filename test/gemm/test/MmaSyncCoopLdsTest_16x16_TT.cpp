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

// Test params for 16 x 16 TT kernels
struct TestParams16x16TT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::row_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL + double
    // Block Sizes: 16 x 16 x BlockK
    // Layouts: TT
    using Types =
        typename Concat<typename Base::TestTypesIOC, typename Base::TestTypeDouble>::Result;
    using BlockSizes = typename Base::TestBlockSizes16x16;
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
class MmaSyncCoopLdsTest16x16TT : public MmaSyncCoopLdsTest
{
};

TEST_P(MmaSyncCoopLdsTest16x16TT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncCoopLdsTest16x16TT,
                         ::testing::Combine(::testing::ValuesIn(TestParams16x16TT::kernels()),
                                            ::testing::ValuesIn(TestParams16x16TT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams16x16TT::problemSizes()),
                                            ::testing::ValuesIn(TestParams16x16TT::alphas()),
                                            ::testing::ValuesIn(TestParams16x16TT::betas())));
