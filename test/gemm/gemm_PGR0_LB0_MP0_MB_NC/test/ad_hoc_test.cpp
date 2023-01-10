/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#include "test/test_includes.hpp"

namespace rocwmma
{

    struct TestParams : public CommonTestParams
    {
        using Base = CommonTestParams;

        // Types: ALL + double
        // Block Sizes: 16 x 16 x BlockK
        // Layouts: NT
        using Types      = std::tuple<std::tuple<float16_t, float32_t, float32_t>>;
        using BlockSizes = std::tuple<std::tuple<I<16>, I<16>, I<16>>>;
        using Layouts    = std::tuple<
            std::tuple<col_major, row_major, row_major>>; //typename Base::TestLayoutsNT;
        using BlocksXY     = std::tuple<std::tuple<I<2>, I<2>>>;
        using KernelParams = typename CombineLists<Types, BlockSizes, Layouts, BlocksXY>::Result;

        // Assemble the kernel generator
        using GeneratorImpl   = KernelGeneratorImpl;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            return {
                //{warpSize, 1},
                {warpSize * 2, 2},
                //{warpSize, 4}, {warpSize * 2, 1}, {warpSize * 2, 2}, {warpSize * 4, 1}
            };
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return {
                //{64, 64, 1024},
                //         {32, 64, 1024},
                // {64, 32, 1024},
                // {256, 256, 1024},
                //{1024, 1024, 1024},
                {64, 64, 64},
                // {128, 128, 128},
                //{2048, 2048, 2048},
                //{8192, 8192, 8192}

            };
        }
    };

} // namespace rocwmma

ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE_NO_WARMUP(Gemm_PGR0_LB0_MP0_MB_NC,
                                               AdHocTest,
                                               rocwmma::TestParams);
