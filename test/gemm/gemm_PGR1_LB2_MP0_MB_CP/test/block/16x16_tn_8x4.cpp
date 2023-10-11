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

#include "test/test_includes.hpp"

namespace rocwmma
{

    ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS(TestParams,
                                             CommonTestParams,
                                             KernelGeneratorImpl,
                                             TestTypesMedium,
                                             TestBlockSizes16x16TinyBlockK,
                                             TestLayoutsTN,
                                             TestLdsDataLayouts,
                                             TestGemmConfigsBlockLevel,
                                             TestBlocks8x4);

} // namespace rocwmma

// Instantiate kernels as a test suite
ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE(Gemm_PGR1_LB2_MP0_MB_CP,
                                     BLK_16x16_TN_8x4,
                                     rocwmma::TestParams);
