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
    ///
    /// Generalized kernel params for most tests
    ///
    struct CommonTestParams
    {
        ///
        /// Testing types as Input/Output/Compute (IOC)
        ///

        // Native int8
        using TestTypesI8 = std::tuple<
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<int8_t, int8_t, int32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<int8_t, int32_t, int32_t>>;

        // Non-native bfloat16_t
        using TestTypesBF16 = std::tuple<
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<bfloat16_t, bfloat16_t, bfloat16_t>,
            std::tuple<bfloat16_t, bfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<bfloat16_t, float32_t, float32_t>>;

        // Native f16
        using TestTypesF16 = std::tuple<
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<float16_t, float16_t, float16_t>,
            std::tuple<float16_t, float16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<float16_t, float32_t, float32_t>>;

        // Non-native hfloat16_t (i.e. __half)
        using TestTypesH16 = std::tuple<
#if defined(ROCWMMA_EXTENDED_TESTS)
            std::tuple<hfloat16_t, hfloat16_t, hfloat16_t>,
            std::tuple<hfloat16_t, hfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<hfloat16_t, float32_t, float32_t>>;

        // Native single f32
        using TestTypesF32 = std::tuple<std::tuple<float32_t, float32_t, float32_t>>;

        // Native double f64
        using TestTypesF64 = std::tuple<std::tuple<float64_t, float64_t, float64_t>>;

        // Aggregate types <= 8 bit
        using TestTypesTiny = TestTypesI8;

        // Aggregate types <= 16 bit
        using TestTypesSmall =
            typename Concat<TestTypesI8, TestTypesBF16, TestTypesF16, TestTypesH16>::Result;

        // Aggregate types <= 32 bit
        using TestTypesMedium = typename Concat<TestTypesSmall, TestTypesF32>::Result;

        // Aggregate types <= 64 bit
        using TestTypesLarge = typename Concat<TestTypesMedium, TestTypesF64>::Result;

        // 16 x 16 supports up to f64
        using TestTypes16x16 = TestTypesLarge;

        // 32 x 32 supports up to f32
        using TestTypes32x32 = TestTypesMedium;

        ///
        /// Data Layout types: col_major (N) or row_major (T)
        /// Matrices layouts: A / B / C / D
        /// Lds Layouts: Lds data
        ///

        // Supported generalized data layouts
        using TestDataLayouts = std::tuple<row_major, col_major>;

        // Lds data layouts
        using TestLdsDataLayouts = TestDataLayouts;

        ///
        /// Aggregate data layout combinations A / B / CD
        /// Note: for the following sets, assume C = D = col_major
        /// Extended tests will test both col_major and row_major.
        ///

        using TestLayoutsNN =
#if defined(ROCWMMA_EXTENDED_TESTS)
            typename CombineOne<std::tuple<col_major, col_major>, TestDataLayouts>::Result;
#else
            std::tuple<col_major, col_major, col_major>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsNT =
#if defined(ROCWMMA_EXTENDED_TESTS)
            typename CombineOne<std::tuple<col_major, row_major>, TestDataLayouts>::Result;
#else
            std::tuple<col_major, row_major, col_major>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsTN =
#if defined(ROCWMMA_EXTENDED_TESTS)
            typename CombineOne<std::tuple<row_major, col_major>, TestDataLayouts>::Result;
#else
            std::tuple<row_major, col_major, col_major>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsTT =
#if defined(ROCWMMA_EXTENDED_TESTS)
            typename CombineOne<std::tuple<row_major, row_major>, TestDataLayouts>::Result;
#else
            std::tuple<row_major, row_major, col_major>;
#endif // ROCWMMA_EXTENDED_TESTS

        ///
        /// MFMA block sizes
        /// BlockK variances for particular BlockM, BlockN
        ///

        // Aggregate combinations BlockK == 16
        using TestBlockSizes16x16TinyBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>>;

        // Aggregate combinations BlockK <= 32
        using TestBlockSizes16x16SmallBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                          ,
                                                          std::tuple<I<16>, I<16>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 64
        using TestBlockSizes16x16MediumBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                           std::tuple<I<16>, I<16>, I<32>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                           ,
                                                           std::tuple<I<16>, I<16>, I<64>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                           >;

        // Aggregate combinations BlockK <= 128
        using TestBlockSizes16x16LargeBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                          std::tuple<I<16>, I<16>, I<32>>,
                                                          std::tuple<I<16>, I<16>, I<64>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                          ,
                                                          std::tuple<I<16>, I<16>, I<128>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 256
        using TestBlockSizes16x16HugeBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                         std::tuple<I<16>, I<16>, I<32>>,
                                                         std::tuple<I<16>, I<16>, I<64>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                         ,
                                                         std::tuple<I<16>, I<16>, I<128>>,
                                                         std::tuple<I<16>, I<16>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                         >;
        // Aggregate combinations BlockK == 8
        using TestBlockSizes32x32TinyBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>>;

        // Aggregate combinations BlockK <= 16
        using TestBlockSizes32x32SmallBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                          ,
                                                          std::tuple<I<32>, I<32>, I<16>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 32
        using TestBlockSizes32x32MediumBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                           std::tuple<I<32>, I<32>, I<16>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                           ,
                                                           std::tuple<I<32>, I<32>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                           >;

        // Aggregate combinations BlockK <= 64
        using TestBlockSizes32x32LargeBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                          std::tuple<I<32>, I<32>, I<16>>,
                                                          std::tuple<I<32>, I<32>, I<32>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                          ,
                                                          std::tuple<I<32>, I<32>, I<64>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 128
        using TestBlockSizes32x32HugeBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                         std::tuple<I<32>, I<32>, I<16>>,
                                                         std::tuple<I<32>, I<32>, I<32>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                         ,
                                                         std::tuple<I<32>, I<32>, I<64>>,
                                                         std::tuple<I<32>, I<32>, I<128>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                         >;

        using TestBlockSizes16x16 = TestBlockSizes16x16HugeBlockK;
        using TestBlockSizes32x32 = TestBlockSizes32x32HugeBlockK;

        ///
        /// Per-wave output block coverage
        ///
        using TestBlocks1x1 = std::tuple<std::tuple<I<1>, I<1>>>;
        using TestBlocks2x1 = std::tuple<std::tuple<I<2>, I<1>>>;
        using TestBlocks1x2 = std::tuple<std::tuple<I<1>, I<2>>>;
        using TestBlocks2x2 = std::tuple<std::tuple<I<2>, I<2>>>;
        using TestBlocks2x4 = std::tuple<std::tuple<I<2>, I<4>>>;
        using TestBlocks4x2 = std::tuple<std::tuple<I<4>, I<2>>>;
        using TestBlocks4x4 = std::tuple<std::tuple<I<4>, I<4>>>;
        using TestBlocks4x8 = std::tuple<std::tuple<I<4>, I<8>>>;
        using TestBlocks8x4 = std::tuple<std::tuple<I<8>, I<4>>>;
        using TestBlocks8x8 = std::tuple<std::tuple<I<8>, I<8>>>;

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
                // clang-format off
                // Don't benchmark wg less than 4 waves by default
#if defined(ROCWMMA_VALIDATION_TESTS) || defined(ROCWMMA_EXTENDED_TESTS)
                {64, 1}, // 1 wave
                {64, 2}, {128, 1}, // 2 wave
#endif // ROCWMMA_VALIDATION_TESTS
                {64, 4}, {128, 2}, // 4 wave
                {256, 1}  // 4 wave
                // clang-format on
            };
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {

            return
            {
                // clang-format off
                {64, 64, 1024},
                {32, 64, 1024},
                {64, 32, 1024},
                {256, 256, 1024},
                {2048, 64, 1024},
                {64, 2048, 1024},
                {1024, 1024, 1024},
                // Skip validation on larger sizes
                // due to very slow.
#if !defined(ROCWMMA_VALIDATION_TESTS)
                {2048, 2048, 2048},
                {2560, 2560, 2560},
                {3072, 3072, 3072},
                {3584, 3584, 3584},
                {4096, 4096, 4096},
                {5120, 5120, 5120},
#ifdef ROCWMMA_EXTENDED_TESTS
                {6144, 6144, 6144},
                {7168, 7168, 7168},
                {8192, 8192, 8192},
#endif // ROCWMMA_EXTENDED_TESTS
#endif // !ROCWMMA_VALIDATION_TESTS \
    // clang-format on
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

///
/// Test suite parameters definition
/// @params
/// test_params_name : name of the resulting class
/// common_base_params : base parameter class holding common symbols
/// kernel_generator_impl : kernel generator implementation class
/// ... (__VA_ARGS__) : a list of kernel parameters that will be combined to create a set of test kernels.
///
#define ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS(test_params_name, common_base_params, kernel_generator_impl, ...)  \
    struct test_params_name : public common_base_params                                                            \
    {                                                                                                              \
        /* Use combinatorial logic to generate a set of kernel params from the input. */                           \
        using KernelParams = typename CombineLists<__VA_ARGS__>::Result;                                           \
        using KernelGenerator = KernelGenerator<KernelParams, kernel_generator_impl>;                              \
                                                                                                                   \
        /* Sanity check to make sure the generator produces kernels expected by the test interface */              \
        static_assert(std::is_same<typename kernel_generator_impl::ResultT,                                        \
                                   typename common_base_params::KernelT>::value,                                   \
                  "Kernels from this generator do not match testing interface");                                   \
                                                                                                                   \
        /* Generate the set of kernels to be tested */                                                             \
        static inline typename KernelGenerator::ResultT kernels()                                                  \
        {                                                                                                          \
            return KernelGenerator::generate();                                                                    \
        }                                                                                                          \
    };

///
/// Test suite instantiation (gtest integration)
/// @params
/// test_suite_prefix: context describing the test suite (e.g. gemm_tests)
/// test_suite_name: name of the specific test suite (e.g. gemm_kernel_5_NN_Layout)
/// test_interface: base gtest interface class
/// test_invoke: name of the test function to invoke on the test suite
/// test_param_triage: triage of parameters delivered to tests (e.g macro to match test_interface with runtime params)
/// test_params: testing parameters used to generate the test suite
///
#define ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix, test_suite_name, test_interface, test_invoke, test_param_triage, test_params) \
    class test_suite_name : public test_interface                                    \
    {                                                                                \
    };                                                                               \
                                                                                     \
    TEST_P(test_suite_name, test_invoke)                                             \
    {                                                                                \
        this->test_invoke();                                                         \
    }                                                                                \
                                                                                     \
    INSTANTIATE_TEST_SUITE_P(                                                        \
        test_suite_prefix,                                                           \
        test_suite_name,                                                             \
        test_param_triage(test_params));

///
/// Triage of test parameters, specific to GEMM gtests.
/// Uses GTest combinatorial function to instantiate all possible
/// combinations of given parameters from each context.
/// @params
/// test_params : the class generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS,
/// which fulfills the rocwmma::GemmTest interface.
///
#define ROCWMMA_GEMM_GTEST_PARAM_TRIAGE(test_params)                                 \
        ::testing::Combine(::testing::ValuesIn(test_params::kernels()),              \
                           ::testing::ValuesIn(test_params::threadBlocks()),         \
                           ::testing::ValuesIn(test_params::problemSizes()),         \
                           ::testing::ValuesIn(test_params::alphas()),               \
                           ::testing::ValuesIn(test_params::betas()))

///
/// Specific to GEMM gtest interface of rocwmma::GemmTest
/// @params
/// test_suite_prefix = used as the general test context (e.g. gemm_kernel_tests)
/// test_suite_name = specific test context (e.g. gemm_my_kernel_NN_32x32_2x1)
/// test_params = the object generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS
/// Note: The rocwmma::GemmTest interface is paired here explicitly with the
/// ROCWMMA_GEMM_GTEST_PARAM_TRIAGE macro to ensure matching of gtest parameters.
/// Invokes the RunKernel() function in rocwmma::GemmTest object.
///
#define ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE(test_suite_prefix, test_suite_name, test_params) \
    ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix,                  \
                                    test_suite_name,                    \
                                    rocwmma::GemmTest,                  \
                                    RunKernel,                          \
                                    ROCWMMA_GEMM_GTEST_PARAM_TRIAGE,    \
                                    test_params)

///
/// Specific to GEMM gtest interface of rocwmma::GemmTest
/// @params
/// test_suite_prefix = used as the general test context (e.g. gemm_kernel_tests)
/// test_suite_name = specific test context (e.g. gemm_my_kernel_NN_32x32_2x1)
/// test_params = the object generated by ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS
/// Note: The rocwmma::GemmTest interface is paired here explicitly with the
/// ROCWMMA_GEMM_GTEST_PARAM_TRIAGE macro to ensure matching of gtest parameters.
/// Invokes the RunKernelWithoutWarmup() function in rocwmma::GemmTest object.
///
#define ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE_NO_WARMUP(test_suite_prefix, test_suite_name, test_params) \
    ROCWMMA_INSTANTIATE_GTEST_SUITE(test_suite_prefix,                  \
                                    test_suite_name,                    \
                                    rocwmma::GemmTest,                  \
                                    RunKernelWithoutWarmup,             \
                                    ROCWMMA_GEMM_GTEST_PARAM_TRIAGE,    \
                                    test_params)


#endif // ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP
