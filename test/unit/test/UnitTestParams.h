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

#ifndef WMMA_UNIT_UNIT_TEST_PARAMS_H
#define WMMA_UNIT_UNIT_TEST_PARAMS_H

#include <tuple>
#include <vector>

#include "Common.hpp"
#include "KernelGenerator.h"
#include "Types.h"
#include "UnitKernelBase.h"

struct UnitTestParams
{
    ///
    /// Compile-time params used with KernelGenerator to
    /// instantiate kernel objects
    ///

    // Testing types as Input/Output/Compute (IOC)
    using TestTypesIOC = std::tuple<
        // Native int8
        float32_t,
        float16_t,
        hfloat16_t,
        int8_t,
        int32_t,
        uint8_t,
        uint32_t>;

    // Native double
    using TestTypeDouble = float64_t;

    ///
    /// Grouped compile time kernel parameters
    ///

    // 16 x 16 has support for double types
    using TestTypes16x16 = typename Concat<TestTypesIOC, TestTypeDouble>::Result;

    // 32 x 32 does not support double types
    using TestTypes32x32 = TestTypesIOC;

    // BlockK variances for particular BlockM, BlockN
    using TestBlockSizes16x16 = std::tuple<std::tuple<I<16>, I<16>>,
                                           std::tuple<I<16>, I<16>>,
                                           std::tuple<I<16>, I<16>>,
                                           std::tuple<I<16>, I<16>>,
                                           std::tuple<I<16>, I<16>>>;

    using TestBlockSizes32x32 = std::tuple<std::tuple<I<32>, I<32>>,
                                           std::tuple<I<32>, I<32>>,
                                           std::tuple<I<32>, I<32>>,
                                           std::tuple<I<32>, I<32>>,
                                           std::tuple<I<32>, I<32>>>;

    // Layout groupings
    using TestLayoutsN = col_major;
    using TestLayoutsT = row_major;

    ///
    /// Run-time kernel argument parameters
    ///

    // Types of parameters
    using KernelT      = std::shared_ptr<KernelI>; // Kernel test interface
    using ThreadBlockT = std::pair<int64_t, int64_t>;
    using ProblemSizeT = std::pair<int64_t, int64_t>;
    using Param1T      = float64_t;
    using Param2T      = float64_t;

    static inline std::vector<ThreadBlockT> threadBlocks()
    {
        return {{64, 1}, {64, 2}, {64, 4}, {128, 1}, {128, 2}, {256, 1}};
    }

    static inline std::vector<ProblemSizeT> problemSizes()
    {
        return {{64, 64},
                {32, 64},
                {64, 32},
                {256, 256},
                {2048, 64},
                {64, 2048},
                {1024, 1024}
#ifndef WMMA_VALIDATE_TESTS
                ,
                {2048, 2048},
                {2560, 2560},
                {3072, 3072},
                {3584, 3584},
                {4096, 4096},
                {5120, 5120},
                {6144, 6144},
                {7168, 7168},
                {8192, 8192}
#endif // WMMA_VALIDATE_TESTS
        };
    }

    static inline std::vector<Param1T> param1s()
    {
        return {0.0};
    }

    static inline std::vector<Param2T> param2s()
    {
        return {0.0};
    }
};

#endif // WMMA_UNIT_UNIT_TEST_PARAMS_H
