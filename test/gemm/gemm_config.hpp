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
#ifndef GEMM_CONFIG_HPP
#define GEMM_CONFIG_HPP

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
        namespace BlockLevel
        {
            struct LdsNT
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
                          uint32_t BlocksY,
                          uint32_t TBlockX = 0,
                          uint32_t TBlockY = 0>
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
                                                                       BlocksY,
                                                                       TBlockX,
                                                                       TBlockY>;

                template <typename GlobalMapping, typename LayoutLds>
                using LdsMapping = LocalMapping::LdsMappingNT<GlobalMapping, LayoutLds>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerA = typename Schedule::SameRowFwd<TBlockX, TBlockY>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerB = typename Schedule::SameColFwd<TBlockX, TBlockY>;

                template <typename GlobalMapping,
                          typename LdsMapping,
                          typename CoopSchedulerA,
                          typename CoopSchedulerB>
                using GemmDriver
                    = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
            };

            struct LdsTN
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
                          uint32_t BlocksY,
                          uint32_t TBlockX = 0,
                          uint32_t TBlockY = 0>
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
                                                                       BlocksY,
                                                                       TBlockX,
                                                                       TBlockY>;

                template <typename GlobalMapping, typename LayoutLds>
                using LdsMapping = LocalMapping::LdsMappingTN<GlobalMapping, LayoutLds>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerA = typename Schedule::SameRowFwd<TBlockX, TBlockY>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerB = typename Schedule::SameColFwd<TBlockX, TBlockY>;

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
                          uint32_t BlocksY,
                          uint32_t TBlockX = 0,
                          uint32_t TBlockY = 0>
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
                                                                       BlocksY,
                                                                       TBlockX,
                                                                       TBlockY>;

                template <typename GlobalMapping, typename LayoutLds>
                using LdsMapping = LocalMapping::LdsMappingRF<GlobalMapping, LayoutLds>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerA = typename Schedule::SameRowFwd<TBlockX, TBlockY>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerB = typename Schedule::SameColFwd<TBlockX, TBlockY>;

                template <typename GlobalMapping,
                          typename LdsMapping,
                          typename CoopSchedulerA,
                          typename CoopSchedulerB>
                using GemmDriver
                    = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
            };

        } // BlockLevel
        namespace WaveLevel
        {
            struct LdsNT
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
                          uint32_t BlocksY,
                          uint32_t TBlockX = 0,
                          uint32_t TBlockY = 0>
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
                                                                      BlocksY,
                                                                      TBlockX,
                                                                      TBlockY>;

                template <typename GlobalMapping, typename LayoutLds>
                using LdsMapping = LocalMapping::LdsMappingNT<GlobalMapping, LayoutLds>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerA = typename Schedule::SameRowFwd<TBlockX, TBlockY>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerB = typename Schedule::SameColFwd<TBlockX, TBlockY>;

                template <typename GlobalMapping,
                          typename LdsMapping,
                          typename CoopSchedulerA,
                          typename CoopSchedulerB>
                using GemmDriver
                    = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
            };

            struct LdsTN
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
                          uint32_t BlocksY,
                          uint32_t TBlockX = 0,
                          uint32_t TBlockY = 0>
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
                                                                      BlocksY,
                                                                      TBlockX,
                                                                      TBlockY>;

                template <typename GlobalMapping, typename LayoutLds>
                using LdsMapping = LocalMapping::LdsMappingTN<GlobalMapping, LayoutLds>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerA = typename Schedule::SameRowFwd<TBlockX, TBlockY>;

                template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
                using CoopSchedulerB = typename Schedule::SameColFwd<TBlockX, TBlockY>;

                template <typename GlobalMapping,
                          typename LdsMapping,
                          typename CoopSchedulerA,
                          typename CoopSchedulerB>
                using GemmDriver
                    = GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;
            };

        } // namespace WaveLevel

    } // namespace CooperativeGemm

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::BlockLevel::LdsNT>()
    {
        return "Block_LdsNT";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::BlockLevel::LdsTN>()
    {
        return "Block_LdsTN";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::BlockLevel::LdsRF>()
    {
        return "Block_LdsRF";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::WaveLevel::LdsNT>()
    {
        return "Wave_LdsNT";
    }

    template <>
    constexpr const char* dataTypeToString<typename CooperativeGemm::WaveLevel::LdsTN>()
    {
        return "Wave_LdsTN";
    }

} // namespace rocwmma

#endif // GEMM_CONFIG_HPP
