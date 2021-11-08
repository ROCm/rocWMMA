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

#include <type_traits>

#include "KernelGenerator.h"
#include "detail/ColLayout.h"
#include "test/UnitTest.h"

struct TestParams : public UnitTestParams
{
    using Base = UnitTestParams;

    // Types: ALL + double
    // Block Sizes: 16 x 16 x BlockK
    // Layouts: T
    using Types        = typename Base::TestTypes16x16;
    using BlockSizes   = typename Base::TestBlockSizes16x16;
    using Layouts      = typename Base::TestLayoutsT;
    using KernelParams = typename CombineLists<Types, BlockSizes, Layouts>::Result;

    // Assemble the kernel generator
    // Kernel: ColLayout
    using GeneratorImpl   = ColLayoutGenerator;
    using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

    // Sanity check for kernel generator
    static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                  "Kernels from this generator do not match testing interface");

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class ColLayoutTest16x16T : public UnitTest
{
};

TEST_P(ColLayoutTest16x16T, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(KernelTests,
                         ColLayoutTest16x16T,
                         ::testing::Combine(::testing::ValuesIn(TestParams::kernels()),
                                            ::testing::ValuesIn(TestParams::threadBlocks()),
                                            ::testing::ValuesIn(TestParams::problemSizes()),
                                            ::testing::ValuesIn(TestParams::param1s()),
                                            ::testing::ValuesIn(TestParams::param2s())));
