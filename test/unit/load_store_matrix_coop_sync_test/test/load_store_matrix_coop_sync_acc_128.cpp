/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2024 Advanced Micro Devices, Inc.
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

#include "detail/load_store_matrix_coop_sync.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{

    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        // Types: Base IOC
        // Block Sizes: 128 x BlockN
        // Layouts: N, T
        using Types        = typename Base::TestTypesIOC;
        using BlockSizes   = std::tuple<std::tuple<I<128>, I<8>>, std::tuple<I<128>, I<16>>>;
        using Layouts      = typename Base::TestLayoutsAll;
        using KernelParams = typename CombineLists<Types, BlockSizes, Layouts>::Result;

        // Assemble the kernel generator
        // Kernel: LoadStoreMatrixCoopSyncAcc
        using GeneratorImpl   = LoadStoreMatrixCoopSyncGeneratorAcc;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }

        static inline std::vector<Base::Param1T> param1s()
        {
            return {0.0, 1.0}; // Split by waves in same rol and col
        }

        static inline std::vector<Base::Param2T> param2s()
        {
            return {
                0.0,
                1.0,
                2.0,
                3.0 // 1 - 4 waves
#ifdef ROCWMMA_EXTENDED_TESTS
                ,
                4.0,
                5.0,
                6.0,
                7.0 // 8 waves
#endif // ROCWMMA_EXTENDED_TESTS
            };
        }
    };

} // namespace rocwmma

// Test suite for unique parameterization
class LoadStoreMatrixSyncCoopAccTest128 : public rocwmma::UnitTest
{
};

TEST_P(LoadStoreMatrixSyncCoopAccTest128, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    LoadStoreMatrixSyncCoopAccTest128,
    ::testing::Combine(::testing::ValuesIn(rocwmma::TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::TestParams::param2s())));
