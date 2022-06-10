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
#ifndef LDS_MAPPING_UTIL_HPP
#define LDS_MAPPING_UTIL_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

#include "gemm_coop_schedule.hpp"
#include "gemm_driver.hpp"
#include "gemm_global_mapping.hpp"
#include "gemm_local_mapping.hpp"

namespace rocwmma
{
    namespace CooperativeGemm
    {
        struct LdsKW
        {
            template <uint32_t BlockM,
                      uint32_t BlockN,
                      uint32_t BlockK,
                      typename InputT,
                      typename OutputT,
                      typename ComputeT,
                      typename LayoutA,
                      typename LayoutB,
                      typename LayoutC,
                      typename LayoutD,
                      uint32_t BlocksX,
                      uint32_t BlocksY>
            using GlobalMapping = GlobalMapping::WaveLevelMapping<BlockM,
                                                                  BlockN,
                                                                  BlockK,
                                                                  InputT,
                                                                  OutputT,
                                                                  ComputeT,
                                                                  LayoutA,
                                                                  LayoutB,
                                                                  LayoutC,
                                                                  LayoutD,
                                                                  BlocksX,
                                                                  BlocksY>;

            template <typename GlobalMapping, typename LayoutLds>
            using LdsMapping = LocalMapping::LdsMappingNT<GlobalMapping, LayoutLds>;

            using CoopSchedulerA = typename Schedule::SameRowFwd;
            using CoopSchedulerB = typename Schedule::SameColFwd;

            template <typename GlobalMapping,
                      typename LdsMapping,
                      typename CoopSchedulerA,
                      typename CoopSchedulerB>
            using GemmDriver
                = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
        };

        struct LdsKH
        {
            template <uint32_t BlockM,
                      uint32_t BlockN,
                      uint32_t BlockK,
                      typename InputT,
                      typename OutputT,
                      typename ComputeT,
                      typename LayoutA,
                      typename LayoutB,
                      typename LayoutC,
                      typename LayoutD,
                      uint32_t BlocksX,
                      uint32_t BlocksY>
            using GlobalMapping = GlobalMapping::WaveLevelMapping<BlockM,
                                                                  BlockN,
                                                                  BlockK,
                                                                  InputT,
                                                                  OutputT,
                                                                  ComputeT,
                                                                  LayoutA,
                                                                  LayoutB,
                                                                  LayoutC,
                                                                  LayoutD,
                                                                  BlocksX,
                                                                  BlocksY>;

            template <typename GlobalMapping, typename LayoutLds>
            using LdsMapping = LocalMapping::LdsMappingTN<GlobalMapping, LayoutLds>;

            using CoopSchedulerA = typename Schedule::SameRowFwd;
            using CoopSchedulerB = typename Schedule::SameColFwd;

            template <typename GlobalMapping,
                      typename LdsMapping,
                      typename CoopSchedulerA,
                      typename CoopSchedulerB>
            using GemmDriver
                = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
        };

        struct LdsRF
        {
            template <uint32_t BlockM,
                      uint32_t BlockN,
                      uint32_t BlockK,
                      typename InputT,
                      typename OutputT,
                      typename ComputeT,
                      typename LayoutA,
                      typename LayoutB,
                      typename LayoutC,
                      typename LayoutD,
                      uint32_t BlocksX,
                      uint32_t BlocksY>
            using GlobalMapping = GlobalMapping::BlockLevelMapping<BlockM,
                                                                   BlockN,
                                                                   BlockK,
                                                                   InputT,
                                                                   OutputT,
                                                                   ComputeT,
                                                                   LayoutA,
                                                                   LayoutB,
                                                                   LayoutC,
                                                                   LayoutD,
                                                                   BlocksX,
                                                                   BlocksY>;

            template <typename GlobalMapping, typename LayoutLds>
            using LdsMapping = LocalMapping::LdsMappingRF<GlobalMapping, LayoutLds>;

            using CoopSchedulerA = typename Schedule::SameRowFwd;
            using CoopSchedulerB = typename Schedule::SameColFwd;

            template <typename GlobalMapping,
                      typename LdsMapping,
                      typename CoopSchedulerA,
                      typename CoopSchedulerB>
            using GemmDriver
                = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
        };

    } // namespace CooperativeGemm

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::LdsKW>()
    {
        return "LdsKW";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::LdsKH>()
    {
        return "LdsKH";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::LdsRF>()
    {
        return "LdsRF";
    }

} // namespace rocwmma

#endif // LDS_MAPPING_UTIL_HPP
