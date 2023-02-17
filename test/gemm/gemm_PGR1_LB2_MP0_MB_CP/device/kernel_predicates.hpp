/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_GEMM_TEST_DEVICE_PREDICATES
#define ROCWMMA_GEMM_TEST_DEVICE_PREDICATES

#include "gemm_test_traits.hpp"

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
} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_PREDICATES
