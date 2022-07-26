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

#include "test/test_includes.hpp"

namespace rocwmma
{

    struct TestParams : public CooperativeTestParams
    {
        /* Use combinatorial logic to generate a set of kernel params from the input. */
        using KernelParams    = typename CombineLists<TestTypesSmall,
                                                   TestBlockSizes32x32LargeMT,
                                                   TestLayoutsTN,
                                                   TestLdsDataLayouts,
                                                   TestGemmConfigsWaveLevel,
                                                   TestBlocks4x4>::Result;
        using KernelGenerator = KernelGenerator<KernelParams, KernelGeneratorImplWaveLevel>;

        /* Sanity check to make sure the generator produces kernels expected by the test interface */
        static_assert(std::is_same<typename KernelGeneratorImplWaveLevel::ResultT,
                                   typename CooperativeTestParams::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        /* Generate the set of kernels to be tested */
        static inline typename KernelGenerator::ResultT kernels();
    };

} // namespace rocwmma

#if __gfx908__

// TODO: Cannot build gfx90a version of this test due to compiler errors.
// Build only for gfx908, but MUST skip runtime tests of this size for gfx90a

namespace rocwmma
{
    inline typename TestParams::KernelGenerator::ResultT TestParams::kernels()
    {
        return KernelGenerator::generate();
    }
}

// Instantiate kernels as a test suite
ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE(Gemm_PGR1_LB2_MP0_MB_CP, WV_32x32_TN_4x4, rocwmma::TestParams);

#endif // __gfx908__
