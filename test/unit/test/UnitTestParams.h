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

#ifndef ROCWMMA_UNIT_UNIT_TEST_PARAMS_H
#define ROCWMMA_UNIT_UNIT_TEST_PARAMS_H

#include <tuple>
#include <vector>

#include "Common.h"
#include "KernelGenerator.h"
#include "UnitKernelBase.h"
#include <WMMA/internal/Types.h>

namespace rocwmma
{

    struct UnitTestParams
    {
        ///
        /// Compile-time params used with KernelGenerator to
        /// instantiate kernel objects
        ///

        // Testing types as Input/Output/Compute (IOC)
        using TestTypesIOC = std::tuple<bfloat16_t,
                                        float16_t,
                                        hfloat16_t,
                                        float32_t,
                                        int8_t
#ifdef WMMA_EXTENDED_TESTS
                                        ,
                                        int32_t,
                                        uint8_t,
                                        uint32_t
#endif // WMMA_EXTENDED_TESTS
                                        >;

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
                                               std::tuple<I<16>, I<32>>,
                                               std::tuple<I<16>, I<64>>
#ifdef WMMA_EXTENDED_TESTS
                                               ,
                                               std::tuple<I<16>, I<128>>,
                                               std::tuple<I<16>, I<256>>
#endif // WMMA_EXTENDED_TESTS
                                               >;

        using TestBlockSizes32x32 = std::tuple<std::tuple<I<32>, I<8>>,
                                               std::tuple<I<32>, I<16>>,
                                               std::tuple<I<32>, I<32>>,
                                               std::tuple<I<32>, I<64>>
#ifdef WMMA_EXTENDED_TESTS
                                               ,
                                               std::tuple<I<32>, I<128>>,
                                               std::tuple<I<32>, I<256>>
#endif // WMMA_EXTENDED_TESTS
                                               >;

        using TestBlockSizes64 = std::tuple<std::tuple<I<64>, I<8>>,
                                            std::tuple<I<64>, I<16>>,
                                            std::tuple<I<64>, I<32>>,
                                            std::tuple<I<64>, I<64>>
#ifdef WMMA_EXTENDED_TESTS
                                            ,
                                            std::tuple<I<64>, I<128>>,
                                            std::tuple<I<64>, I<256>>
#endif // WMMA_EXTENDED_TESTS
                                            >;

        using TestBlockSizes128 = std::tuple<std::tuple<I<128>, I<8>>,
                                             std::tuple<I<128>, I<16>>,
                                             std::tuple<I<128>, I<32>>,
                                             std::tuple<I<128>, I<64>>
#ifdef WMMA_EXTENDED_TESTS
                                             ,
                                             std::tuple<I<128>, I<128>>,
                                             std::tuple<I<128>, I<256>>
#endif // WMMA_EXTENDED_TESTS
                                             >;

        using TestBlockSizes256 = std::tuple<std::tuple<I<256>, I<8>>,
                                             std::tuple<I<256>, I<16>>,
                                             std::tuple<I<256>, I<32>>,
                                             std::tuple<I<256>, I<64>>
#ifdef WMMA_EXTENDED_TESTS
                                             ,
                                             std::tuple<I<256>, I<128>>,
                                             std::tuple<I<256>, I<256>>
#endif // WMMA_EXTENDED_TESTS
                                             >;

        // Layout groupings
        using TestLayoutsN   = col_major;
        using TestLayoutsT   = row_major;
        using TestLayoutsAll = typename Concat<TestLayoutsN, TestLayoutsT>::Result;

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
            // clang-format off
        return { {64, 1},  // 1 Wave
                 {64, 2}, {128, 1}, // 2 Waves
                 {64, 4}, {128, 2}, {256, 1}, // 4 Waves
#ifdef WMMA_EXTENDED_TESTS
                 {64, 8}, {128, 4}, {256, 2}, {512, 1} // 8 waves
#endif // WMMA_EXTENDED_TESTS
            };
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
        // Test at least all the 1-wave and rectangular sizes
        return { {16, 16},  {16, 32},   {16, 64},   {16, 128},  {16, 256},
                 {32, 8},   {32, 16},   {32, 32},   {32, 64},   {32, 128},   {32, 256},
                 {64, 8},   {64, 16},   {64, 32},   {64, 64},   {64, 128},   {64, 256},
                 {128, 8},  {128, 16},  {128, 32},  {128, 64},  {128, 128},  {128, 256},
                 {256, 8},  {256, 16},  {256, 32},  {256, 64},  {256, 128},  {256, 256},
                 {512, 8},  {512, 16},  {512, 32},  {512, 64},  {512, 128},  {512, 256},  {512, 512},
                 {1024, 8}, {1024, 16}, {1024, 32}, {1024, 64}, {1024, 128}, {1024, 256}, {1024, 512},

#ifdef WMMA_EXTENDED_TESTS
                 {1024, 1024},
                 {2048, 2048},
                 {2560, 2560},
                 {3072, 3072},
                 {3584, 3584},
                 {4096, 4096},
                 {5120, 5120},
                 {6144, 6144},
                 {7168, 7168},
                 {8192, 8192}
#endif // WMMA_EXTENDED_TESTS
        };
            // clang-format on
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

} // namespace rocwmma

#endif // ROCWMMA_UNIT_UNIT_TEST_PARAMS_H
