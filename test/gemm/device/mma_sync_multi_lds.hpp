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

#ifndef ROCWMMA_DEVICE_MMA_SYNC_MULTI_LDS_HPP
#define ROCWMMA_DEVICE_MMA_SYNC_MULTI_LDS_HPP

#include <algorithm>
#include <array>
#include <utility>

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "lds_mapping_util.hpp"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
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
              typename LayoutLds,
              typename GemmConfig,
              uint32_t BlocksX = 1,
              uint32_t BlocksY = 1>
    __global__ void __launch_bounds__(256) mmaSyncMultiLds(uint32_t       m,
                                                           uint32_t       n,
                                                           uint32_t       k,
                                                           InputT const*  a,
                                                           InputT const*  b,
                                                           OutputT const* c,
                                                           OutputT*       d,
                                                           uint32_t       lda,
                                                           uint32_t       ldb,
                                                           uint32_t       ldc,
                                                           uint32_t       ldd,
                                                           ComputeT       alpha,
                                                           ComputeT       beta)
    {
        ///
        /// Assemble the gemm driver from the incoming gemm configuration
        ///
        using GlobalMapping = typename GemmConfig::template GlobalMapping<BlockM,
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

        using LdsMapping = typename GemmConfig::template LdsMapping<GlobalMapping, LayoutLds>;
        using GemmDriver =
            typename GemmConfig::template GemmDriver<GlobalMapping,
                                                     LdsMapping,
                                                     typename GemmConfig::CoopSchedulerA,
                                                     typename GemmConfig::CoopSchedulerB>;

        // Global fragments used in pre-fetching
        using GRFragA = typename GlobalMapping::GRFragA;
        using GRFragB = typename GlobalMapping::GRFragB;

        // Fragments for mfma
        using MfmaFragA   = typename GlobalMapping::MfmaFragA;
        using MfmaFragB   = typename GlobalMapping::MfmaFragB;
        using MfmaFragC   = typename GlobalMapping::MfmaFragC;
        using MfmaFragD   = typename GlobalMapping::MfmaFragD;
        using MfmaFragAcc = typename GlobalMapping::MfmaFragAcc;

        // Mapping utils for each fragment type
        using DataMappingA   = typename GetIOShape_t<MfmaFragA>::DataLayout;
        using DataMappingB   = typename GetIOShape_t<MfmaFragB>::DataLayout;
        using DataMappingC   = typename GetIOShape_t<MfmaFragC>::DataLayout;
        using DataMappingD   = typename GetIOShape_t<MfmaFragD>::DataLayout;
        using DataMappingLds = typename LdsMapping::DataLayout;

        ///
        /// Target starting C / D macro tile matrix coordinate on 2D grid
        ///
        auto matrixCoordC = GlobalMapping::matrixCoordC();

        // Bounds check
        if((std::get<0>(matrixCoordC) + BlocksX * BlockM) > m
           || (std::get<1>(matrixCoordC) + BlocksY * BlockN) > n || (BlockK > k))
        {
            return;
        }

        ///
        /// Setup global addressing offsets
        ///
        auto globalOffsetA = DataMappingA::fromMatrixCoord(GlobalMapping::matrixCoordA(), lda);
        auto globalOffsetB = DataMappingB::fromMatrixCoord(GlobalMapping::matrixCoordB(), ldb);
        auto globalOffsetC = DataMappingC::fromMatrixCoord(matrixCoordC, ldc);
        auto globalOffsetD = DataMappingD::fromMatrixCoord(matrixCoordC, ldd);

        auto kStepOffsetA = DataMappingA::fromMatrixCoord(GlobalMapping::kStepA(), lda);
        auto kStepOffsetB = DataMappingB::fromMatrixCoord(GlobalMapping::kStepB(), ldb);

        ///
        /// Start global prefetch
        ///
        typename GlobalMapping::GRBuffA grBuffA;
        typename GlobalMapping::GRBuffB grBuffB;
        GemmDriver::globalReadCoopA(grBuffA, a + globalOffsetA, lda);
        GemmDriver::globalReadCoopB(grBuffB, b + globalOffsetB, ldb);
        globalOffsetA += kStepOffsetA;
        globalOffsetB += kStepOffsetB;

        ///
        /// Setup LDS addressing
        /// This kernel will use 2 separate LDS blocks
        /// for pipelining in the accumulation loop
        ///
        HIP_DYNAMIC_SHARED(void*, localMemPtr);
        auto  sizeLds  = LdsMapping::sizeLds();
        auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
        auto* ldsPtrHi = ldsPtrLo + std::get<0>(sizeLds) * std::get<1>(sizeLds);

        auto ldlds      = LdsMapping::ldLds();
        auto ldsOffsetA = DataMappingLds::fromMatrixCoord(LdsMapping::matrixCoordA(), ldlds);
        auto ldsOffsetB = DataMappingLds::fromMatrixCoord(LdsMapping::matrixCoordB(), ldlds);

        ///
        /// Write prefetch to local
        ///
        GemmDriver::localWriteCoopA(ldsPtrLo + ldsOffsetA, grBuffA, ldlds);
        GemmDriver::localWriteCoopB(ldsPtrLo + ldsOffsetB, grBuffB, ldlds);

        ///
        /// Initialize accumulation frags
        ///
        typename GlobalMapping::MfmaBuffAcc fragsAcc;
        GemmDriver::fill(fragsAcc, static_cast<ComputeT>(0));

        ///
        /// Synchronize waves and memory
        ///
        GemmDriver::syncWorkgroup();

        ///
        /// Accumulate A * B
        ///
        if(alpha)
        {
            for(int currentK = BlockK; currentK < k; currentK += BlockK)
            {
                typename GlobalMapping::MfmaBuffA fragsA;
                typename GlobalMapping::MfmaBuffB fragsB;

                // Local read mfma frags
                GemmDriver::localReadA(fragsA, ldsPtrLo + ldsOffsetA, ldlds);
                GemmDriver::localReadB(fragsB, ldsPtrLo + ldsOffsetB, ldlds);

                // Start fetching next round of frags
                GemmDriver::globalReadCoopA(grBuffA, a + globalOffsetA, lda);
                GemmDriver::globalReadCoopB(grBuffB, b + globalOffsetB, ldb);

                // Advance offsets to next k step
                globalOffsetA += kStepOffsetA;
                globalOffsetB += kStepOffsetB;

                // accum(A * B)
                GemmDriver::mfma(fragsAcc, fragsA, fragsB, fragsAcc);

                GemmDriver::localWriteCoopA(ldsPtrHi + ldsOffsetA, grBuffA, ldlds);
                GemmDriver::localWriteCoopB(ldsPtrHi + ldsOffsetB, grBuffB, ldlds);

                // Make sure that all waves have finished reading / writing to lds.
                GemmDriver::syncWorkgroup();

                // Swap Lds buffers
                auto* tmp = ldsPtrLo;
                ldsPtrLo  = ldsPtrHi;
                ldsPtrHi  = tmp;
            }
        }

        ///
        /// Start loading C
        ///

        typename GlobalMapping::MfmaBuffC fragsC;
        if(beta)
        {
            GemmDriver::globalReadC(fragsC, c + globalOffsetC, ldc);
        }
        else
        {
            GemmDriver::fill(fragsC, 0);
        }

        ///
        /// Clean up tail A * B
        ///

        typename GlobalMapping::MfmaBuffA fragsA;
        typename GlobalMapping::MfmaBuffB fragsB;

        GemmDriver::localReadA(fragsA, ldsPtrLo + ldsOffsetA, ldlds);
        GemmDriver::localReadB(fragsB, ldsPtrLo + ldsOffsetB, ldlds);
        GemmDriver::mfma(fragsAcc, fragsA, fragsB, fragsAcc);

        ///
        /// D = alpha * accum + beta * C
        ///
        typename GlobalMapping::MfmaBuffD fragsD;
        GemmDriver::uniformFma(fragsD, alpha, fragsAcc, beta, fragsC);
        GemmDriver::globalWriteD(d + globalOffsetD, fragsD, ldd);
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_MMA_SYNC_MULTI_LDS_HPP
