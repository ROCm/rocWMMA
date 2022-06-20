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

#ifndef ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP
#define ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <rocwmma/internal/types.hpp>

#include "common.hpp"
#include "gemm_kernel_base.hpp"
#include "kernel_generator.hpp"

namespace rocwmma
{
    namespace CooperativeGemm
    {
        namespace BlockLevel
        {
            class LdsNT;
            class LdsTN;
            class LdsRF;

        } // namespace BlockLevel

        namespace WaveLevel
        {
            class LdsNT;
            class LdsTN;

        } // namespace WaveLevel

        namespace WorkgroupLevel
        {
            class LdsNT;
            class LdsTN;

        } // namespace WaveLevel

    } // namespace CooperativeGemm

    struct CommonTestParams
    {
        ///
        /// Compile-time params used with KernelGenerator to
        /// instantiate kernel objects
        ///

        // Testing types as Input/Output/Compute (IOC)
        using TestTypesSmall = std::tuple<
        // Non-native bfloat16_t

#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<bfloat16_t, bfloat16_t, bfloat16_t>,
            std::tuple<bfloat16_t, bfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<bfloat16_t, float32_t, float32_t>,

        // Native fp16
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<float16_t, float16_t, float16_t>,
            std::tuple<float16_t, float16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<float16_t, float32_t, float32_t>,

        // Non-native hfloat16_t (i.e. __half)
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<hfloat16_t, hfloat16_t, hfloat16_t>,
            std::tuple<hfloat16_t, hfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<hfloat16_t, float32_t, float32_t>,

        // Native int8
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<int8_t, int8_t, int32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<int8_t, int32_t, int32_t>>;

        // Native single f32
        using TestTypeSingle = std::tuple<std::tuple<float32_t, float32_t, float32_t>>;

        // Native double f64
        using TestTypeDouble = std::tuple<std::tuple<float64_t, float64_t, float64_t>>;

        // Supported layout types
        using TestLayoutTypes = std::tuple<row_major, col_major>;

        using TestLdsLayoutTypes = std::tuple<row_major
#if defined(ROCWMMA_VALIDATION_TESTS) || defined(ROCWMMA_EXTENDED_TESTS)
                                              ,
                                              col_major
#endif // ROCWMMA_EXTENDED_TESTS
                                              >;

        // Supported LDS mappings
        using TestMappingsLds = std::tuple<
#if defined(ROCWMMA_VALIDATION_TESTS) || defined(ROCWMMA_EXTENDED_TESTS)
            // Only validate the block level configs
            //std::tuple<typename CooperativeGemm::BlockLevel::LdsNT>,
            //std::tuple<typename CooperativeGemm::BlockLevel::LdsTN>,
            //std::tuple<typename CooperativeGemm::BlockLevel::LdsRF>,
            std::tuple<typename CooperativeGemm::WaveLevel::LdsNT>,
            std::tuple<typename CooperativeGemm::BlockLevel::LdsRF>,
#endif // ROCWMMA_EXTENDED_TESTS \
    // Benchmark only wave-level as they are more performant
            std::tuple<typename CooperativeGemm::WaveLevel::LdsTN>>;

        using TestGemmConfigsWgLevel = std::tuple<
#if defined(ROCWMMA_VALIDATION_TESTS) || defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<typename CooperativeGemm::WorkgroupLevel::LdsNT>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<typename CooperativeGemm::WorkgroupLevel::LdsTN>>;

        ///
        /// Grouped compile time kernel parameters
        ///

        // Default base set of types to test
        using TestTypesIOC = typename Concat<TestTypesSmall, TestTypeSingle>::Result;

        // 16 x 16 has support for double types
        using TestTypes16x16 = typename Concat<TestTypesIOC, TestTypeDouble>::Result;

        // 32 x 32 does not support double types
        using TestTypes32x32 = TestTypesIOC;

        // BlockK variances for particular BlockM, BlockN
        using TestBlockSizes16x16 = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                               std::tuple<I<16>, I<16>, I<32>>,
                                               std::tuple<I<16>, I<16>, I<64>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                               ,
                                               std::tuple<I<16>, I<16>, I<128>>,
                                               std::tuple<I<16>, I<16>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                               >;

        using TestBlockSizes32x32 = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                               std::tuple<I<32>, I<32>, I<16>>,
                                               std::tuple<I<32>, I<32>, I<32>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                               ,
                                               std::tuple<I<32>, I<32>, I<64>>,
                                               std::tuple<I<32>, I<32>, I<128>>
#endif // ROCWMMA_EXTENDED_TESTS
                                               >;

        // Layout groupings
        using TestLayoutsNN =
            typename CombineOne<std::tuple<col_major, col_major>, TestLayoutTypes>::Result;
        using TestLayoutsNT =
            typename CombineOne<std::tuple<col_major, row_major>, TestLayoutTypes>::Result;
        using TestLayoutsTN =
            typename CombineOne<std::tuple<row_major, col_major>, TestLayoutTypes>::Result;
        using TestLayoutsTT =
            typename CombineOne<std::tuple<row_major, row_major>, TestLayoutTypes>::Result;

        ///
        /// Run-time kernel argument parameters
        ///

        // Types of parameters
        using KernelT      = std::shared_ptr<KernelI>; // Kernel test interface
        using ThreadBlockT = std::pair<int64_t, int64_t>;
        using ProblemSizeT = std::tuple<int64_t, int64_t, int64_t>;
        using AlphaT       = float64_t;
        using BetaT        = float64_t;

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            return
            {
#if defined(ROCWMMA_VALIDATION_TESTS) || defined(ROCWMMA_EXTENDED_TESTS)
                // Don't benchmark wg less than 4 waves
                {64, 1}, // 1 wave
                    {64, 2}, {128, 1}, // 2 wave
#endif // ROCWMMA_VALIDATION_TESTS
                    {64, 4}, {128, 2},
                {
                    256, 1
                }
            }; // 4 wave
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return {
                {64, 64, 1024},
                {32, 64, 1024},
                {64, 32, 1024},
                {256, 256, 1024},
                {2048, 64, 1024},
                {64, 2048, 1024},
                {1024, 1024, 1024},
#ifndef ROCWMMA_VALIDATION_TESTS
                {2048, 2048, 2048},
                {2560, 2560, 2560},
                {3072, 3072, 3072},
                {3584, 3584, 3584},
                {4096, 4096, 4096},
                {5120, 5120, 5120},
#endif // ROCWMMA_VALIDATION_TESTS
#ifdef ROCWMMA_EXTENDED_TESTS
                {6144, 6144, 6144},
                {7168, 7168, 7168},
                {8192, 8192, 8192},
#endif
            };
        }

        static inline std::vector<AlphaT> alphas()
        {
            return {2.0};
        }

        static inline std::vector<BetaT> betas()
        {
            return {2.0};
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP
