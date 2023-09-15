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
              uint32_t WaveSize,
              uint32_t ArchId>
    struct gemm_PGR0_LB0_MP0_SB_NC_guard
    {
        using TestTraits = GemmTestTraits<BlockM,
                                          BlockN,
                                          BlockK,
                                          InputT,
                                          OutputT,
                                          ComputeT,
                                          1u,
                                          1u,
                                          WaveSize,
                                          ArchId>;

    private:
        enum struct Gfx9Predicates : bool
        {
            // Must skip int8 tests on gfx9 for now
            CostABTest
            = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB) <= 256u),
            CostCTest = ((uint32_t)TestTraits::Cost::TileC <= 256u),
            CostDTest = ((uint32_t)TestTraits::Cost::TileD <= 256u),

            Enable = ((bool)TestTraits::IsGfx9 && (bool)TestTraits::IsWave64
                      && CostABTest && CostCTest && CostDTest)
        };

        enum struct Gfx11Predicates : bool
        {
            IsFp16
            = std::is_same<InputT, float16_t>::value || std::is_same<InputT, hfloat16_t>::value,
            IsBf16    = std::is_same<InputT, hip_bfloat16>::value,
            IsInt8    = std::is_same<InputT, int8_t>::value,
            TypesTest = IsFp16 || IsBf16 || IsInt8,

            // AB inputs are duplicated, single buffered
            // C tiles are unpacked.
            CostABTest
            = ((2u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostCTest     = ((2u * (uint32_t)TestTraits::Cost::TileC) <= 256u),
            CostDTest     = ((uint32_t)TestTraits::Cost::TileD <= 256u),
            BlockSizeTest = ((BlockM == 16u) && (BlockN == 16u)),

            Enable = ((bool)TestTraits::IsGfx11 && (bool)TestTraits::IsWave32 && TypesTest
                      && CostABTest && CostCTest && CostDTest && BlockSizeTest)
        };

    public:
        constexpr static bool enable()
        {
            return ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable);
        }
    };
} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_PREDICATES
