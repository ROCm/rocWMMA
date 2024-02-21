/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <tuple>
#include <type_traits>

#include "detail/cross_lane_ops.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{

    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        // Types: Base IOC + double
        using Types = typename Base::TestAllSizeTypes;

        using SwizzleOps = std::tuple<SwizzleImpl::Ops::RotateL8<0>,
                                      SwizzleImpl::Ops::RotateL8<1>,
                                      SwizzleImpl::Ops::RotateL8<2>,
                                      SwizzleImpl::Ops::RotateL8<3>,
                                      SwizzleImpl::Ops::RotateL8<4>,
                                      SwizzleImpl::Ops::RotateL8<5>,
                                      SwizzleImpl::Ops::RotateL8<6>,
                                      SwizzleImpl::Ops::RotateL8<7>>;

        using KernelParams = typename CombineLists<Types, SwizzleOps>::Result;

        // Assemble the kernel generator
        // Kernel: VectorIterator
        using GeneratorImpl   = SwizzleOpsGenerator;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        // Must be TBlockY must be 1.
        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            // clang-format off
            return {
                        {warpSize, 1},
                        {warpSize * 2, 1},
                        {warpSize * 4, 1}
                    };
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
            return {
                        {64, 64},
                        {128, 128},
                        {256, 256}
                    };
            // clang-format on
        }

        // 'prev' values
        static inline std::vector<Param1T> param1s()
        {
            return {5.0};
        }

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

} // namespace rocwmma

// Test suite for unique parameterization
class SwizzleRotateL8Test : public rocwmma::UnitTest
{
};

TEST_P(SwizzleRotateL8Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    CrossLaneOpTests,
    SwizzleRotateL8Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::TestParams::param2s())));
