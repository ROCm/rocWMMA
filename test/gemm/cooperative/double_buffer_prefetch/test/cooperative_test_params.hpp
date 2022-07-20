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

#ifndef ROCWMMA_GEMM_COOPERATIVE_TEST_PARAMS_HPP
#define ROCWMMA_GEMM_COOPERATIVE_TEST_PARAMS_HPP

#include "gemm_common_test_params.hpp"

namespace rocwmma
{
    ///
    /// FWD declarations
    ///

    class MmaSyncMultiLdsGenerator;
    class MmaSyncMultiLdsGenerator;
    class MmaSyncCoopWgGenerator;

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

    ///
    /// Generalized kernel params for most cooperative tests
    ///
    struct CooperativeTestParams : public CommonTestParams
    {

        ///
        /// MFMA block sizes
        /// BlockK variances for particular BlockM, BlockN
        /// Cooperative tests generally have larger Macro Tiles (MT)
        /// therefore to limit register usage we reduce the variation
        /// in BlocK sizes.
        /// Small MT  <= 4x4 blocks
        /// Large MT  > 4x4 blocks
        ///

        using TestBlockSizes16x16SmallMT = std::tuple<std::tuple<I<16>, I<16>, I<16>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                      ,
                                                      std::tuple<I<16>, I<16>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                      >;

        using TestBlockSizes16x16LargeMT = std::tuple<std::tuple<I<16>, I<16>, I<16>>>;

        using TestBlockSizes32x32SmallMT = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                      std::tuple<I<32>, I<32>, I<16>>
#if defined(ROCWMMA_EXTENDED_TESTS)
                                                      ,
                                                      std::tuple<I<32>, I<32>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                      >;

        using TestBlockSizes32x32LargeMT = std::tuple<std::tuple<I<32>, I<32>, I<8>>>;

        ///
        /// Cooperative GEMM configurations
        /// Block, Wave and Workgroup levels
        ///

        using TestGemmConfigsBlockLevel
            = std::tuple<std::tuple<typename CooperativeGemm::BlockLevel::LdsNT>,
                         std::tuple<typename CooperativeGemm::BlockLevel::LdsTN>,
                         std::tuple<typename CooperativeGemm::BlockLevel::LdsRF>>;

        using TestGemmConfigsWaveLevel
            = std::tuple<std::tuple<typename CooperativeGemm::WaveLevel::LdsNT>,
                         std::tuple<typename CooperativeGemm::WaveLevel::LdsTN>>;

        using TestGemmConfigsWgLevel
            = std::tuple<std::tuple<typename CooperativeGemm::WorkgroupLevel::LdsNT>,
                         std::tuple<typename CooperativeGemm::WorkgroupLevel::LdsTN>>;

        ///
        /// Kernel generator impl objects
        ///
        using KernelGeneratorImplBlockLevel = MmaSyncMultiLdsGenerator;
        using KernelGeneratorImplWaveLevel  = MmaSyncMultiLdsGenerator;
        using KernelGeneratorImplWgLevel    = MmaSyncCoopWgGenerator;
    };

} // namespace rocwmma

#endif // ROCWMMA_GEMM_COOPERATIVE_TEST_PARAMS_HPP
