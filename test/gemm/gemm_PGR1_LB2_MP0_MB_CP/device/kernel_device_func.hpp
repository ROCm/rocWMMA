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

#ifndef ROCWMMA_GEMM_TEST_DEVICE_FUNC
#define ROCWMMA_GEMM_TEST_DEVICE_FUNC

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "gemm_config.hpp"
#include "gemm_test_traits.hpp"
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
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct gemm_PGR1_LB2_MP0_MB_CP_guard
    {
        using TestTraits = GemmTestTraits<BlockM,
                                          BlockN,
                                          BlockK,
                                          InputT,
                                          OutputT,
                                          ComputeT,
                                          BlocksX,
                                          BlocksY,
                                          WaveSize,
                                          ArchId>;

    private:
        enum struct GlobalPredicates : bool
        {
            // ThreadblockX must be a multiple of the wave size
            TBlockXMult   = (TBlockX % WaveSize == 0u),
            MaxWaveCount4 = (TBlockX / WaveSize * TBlockY <= 4u),

            Enable = (TBlockXMult && MaxWaveCount4)
        };

        enum struct Gfx9Predicates : bool
        {
            CostABTest
            = ((2u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostAccTest  = ((uint32_t)TestTraits::Cost::TileC <= 256u),
            CostTailTest = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB
                             + 2u * (uint32_t)TestTraits::Cost::TileD)
                            <= 256u),

            // Must skip int8 tests on gfx9 for now
            IsInt8 = std::is_same<int8_t, InputT>::value,

            Enable = ((bool)TestTraits::IsGfx9 && (bool)TestTraits::IsWave64 && !(bool)IsInt8
                      && CostABTest && CostAccTest && CostTailTest)
        };

        enum struct Gfx11Predicates : bool
        {
            IsFp16
            = std::is_same<InputT, float16_t>::value || std::is_same<InputT, hfloat16_t>::value,
            IsBf16    = std::is_same<InputT, hip_bfloat16>::value,
            IsInt8    = std::is_same<InputT, int8_t>::value,
            TypesTest = (IsFp16 || IsBf16) && !IsInt8,

            // AB inputs are duplicated, double buffered
            // Acc tiles are unpacked.
            // Tail requires A, B, C & D tiles + FMA
            CostABTest
            = ((4u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostAccTest   = ((2u * (uint32_t)TestTraits::Cost::TileC) <= 256u),
            CostTailTest  = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB
                             + 2u * (uint32_t)TestTraits::Cost::TileD)
                            <= 256u),
            BlockSizeTest = ((BlockM == 16u) && (BlockN == 16u)),

            Enable = ((bool)TestTraits::IsGfx11 && (bool)TestTraits::IsWave32 && TypesTest
                      && CostABTest && CostAccTest && CostTailTest && BlockSizeTest)
        };

    public:
        constexpr static bool enable()
        {
            return ((bool)GlobalPredicates::Enable
                    && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable));
        }
    };

    ///
    /// Device function GEMM kernel:
    ///
    /// PGR1 = Prefetch Global Read, x1 step prefetch
    /// LB2 = Lds Buffer, x2 buffers
    /// MP0 = Mfma Priority, 0
    /// MB = Multi-block output
    /// CP = Cooperative wave-wise global read
    ///
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
              uint32_t BlocksX                                   = 1,
              uint32_t BlocksY                                   = 1,
              uint32_t TBlockX                                   = 0,
              uint32_t TBlockY                                   = 0,
              typename std::enable_if_t<gemm_PGR1_LB2_MP0_MB_CP_guard<
                  BlockM,
                  BlockN,
                  BlockK,
                  InputT,
                  OutputT,
                  ComputeT,
                  LayoutA,
                  LayoutB,
                  LayoutC,
                  LayoutD,
                  LayoutLds,
                  BlocksX,
                  BlocksY,
                  TBlockX,
                  TBlockY,
                  Constants::AMDGCN_WAVE_SIZE,
                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void __launch_bounds__(256) gemm_PGR1_LB2_MP0_MB_CP(uint32_t       m,
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
                                                                          BlocksY,
                                                                          TBlockX,
                                                                          TBlockY>;

        using LdsMapping     = typename GemmConfig::template LdsMapping<GlobalMapping, LayoutLds>;
        using CoopSchedulerA = typename GemmConfig::template CoopSchedulerA<TBlockX, TBlockY>;
        using CoopSchedulerB = typename GemmConfig::template CoopSchedulerB<TBlockX, TBlockY>;
        using GemmDriver     = typename GemmConfig::
            template GemmDriver<GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB>;

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
        auto matrixCoordC  = GlobalMapping::readCoordC();
        auto waveTileDim   = GlobalMapping::waveTileSizeC();
        auto waveTileBound = matrixCoordC + waveTileDim;

        // Bounds check
        if((get<0>(waveTileBound) > m) || (get<1>(waveTileBound) > n))
        {
            return;
        }

        if(BlockK > k)
        {
            return;
        }

        ///
        /// Setup global addressing offsets in 1D
        ///
        auto globalReadOffsetA  = DataMappingA::fromMatrixCoord(GlobalMapping::readCoordA(), lda);
        auto globalReadOffsetB  = DataMappingB::fromMatrixCoord(GlobalMapping::readCoordB(), ldb);
        auto globalReadOffsetC  = DataMappingC::fromMatrixCoord(GlobalMapping::readCoordC(), ldc);
        auto globalWriteOffsetD = DataMappingD::fromMatrixCoord(GlobalMapping::writeCoordD(), ldd);

        auto kStepOffsetA = DataMappingA::fromMatrixCoord(GlobalMapping::kStepOffsetA(), lda);
        auto kStepOffsetB = DataMappingB::fromMatrixCoord(GlobalMapping::kStepOffsetB(), ldb);

        ///
        /// Start global prefetch
        ///
        typename GlobalMapping::GRBuffA grBuffA;
        typename GlobalMapping::GRBuffB grBuffB;
        GemmDriver::globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
        GemmDriver::globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);
        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        ///
        /// Setup LDS addressing
        /// This kernel will use 2 separate LDS blocks
        /// for pipelining in the accumulation loop
        ///
        HIP_DYNAMIC_SHARED(void*, localMemPtr);
        auto  sizeLds  = LdsMapping::sizeLds();
        auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
        auto* ldsPtrHi = ldsPtrLo + get<0>(sizeLds) * get<1>(sizeLds);

        auto ldlds           = LdsMapping::ldLds();
        auto ldsWriteOffsetA = DataMappingLds::fromMatrixCoord(LdsMapping::writeCoordA(), ldlds);
        auto ldsWriteOffsetB = DataMappingLds::fromMatrixCoord(LdsMapping::writeCoordB(), ldlds);
        auto ldsReadOffsetA  = DataMappingLds::fromMatrixCoord(LdsMapping::readCoordA(), ldlds);
        auto ldsReadOffsetB  = DataMappingLds::fromMatrixCoord(LdsMapping::readCoordB(), ldlds);

        ///
        /// Write prefetch to local
        ///
        GemmDriver::localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldlds);
        GemmDriver::localWriteCoopB(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldlds);

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
        for(auto currentK = BlockK; currentK < k; currentK += BlockK)
        {
            typename GlobalMapping::MfmaBuffA fragsA;
            typename GlobalMapping::MfmaBuffB fragsB;

            // Local read mfma frags
            GemmDriver::localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldlds);
            GemmDriver::localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldlds);

            // Start fetching next round of frags
            GemmDriver::globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
            GemmDriver::globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);

            // Advance offsets to next k step
            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            // accum(A * B)
            GemmDriver::mfma(fragsAcc, fragsA, fragsB, fragsAcc);

            GemmDriver::localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldlds);
            GemmDriver::localWriteCoopB(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldlds);

            // Make sure that all waves have finished reading / writing to lds.
            GemmDriver::syncWorkgroup();

            // Swap Lds buffers
            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;
        }

        ///
        /// Start loading C
        ///

        typename GlobalMapping::MfmaBuffC fragsC;
        GemmDriver::globalReadC(fragsC, c + globalReadOffsetC, ldc);

        ///
        /// Clean up tail A * B
        ///

        typename GlobalMapping::MfmaBuffA fragsA;
        typename GlobalMapping::MfmaBuffB fragsB;

        GemmDriver::localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldlds);
        GemmDriver::localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldlds);
        GemmDriver::mfma(fragsAcc, fragsA, fragsB, fragsAcc);

        ///
        /// D = alpha * accum + beta * C
        ///
        typename GlobalMapping::MfmaBuffD fragsD;
        GemmDriver::uniformFma(fragsD, alpha, fragsAcc, beta, fragsC);
        GemmDriver::globalWriteD(d + globalWriteOffsetD, fragsD, ldd);
    }

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
              uint32_t BlocksX                                   = 1,
              uint32_t BlocksY                                   = 1,
              uint32_t TBlockX                                   = 0,
              uint32_t TBlockY                                   = 0,
              typename std::enable_if_t<!gemm_PGR1_LB2_MP0_MB_CP_guard<
                  BlockM,
                  BlockN,
                  BlockK,
                  InputT,
                  OutputT,
                  ComputeT,
                  LayoutA,
                  LayoutB,
                  LayoutC,
                  LayoutD,
                  LayoutLds,
                  BlocksX,
                  BlocksY,
                  TBlockX,
                  TBlockY,
                  Constants::AMDGCN_WAVE_SIZE,
                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void __launch_bounds__(256) gemm_PGR1_LB2_MP0_MB_CP(uint32_t       m,
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
    }

} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_FUNC
