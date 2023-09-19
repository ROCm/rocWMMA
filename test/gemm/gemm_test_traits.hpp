/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_GEMM_TEST_TRAITS_HPP
#define ROCWMMA_GEMM_TEST_TRAITS_HPP

// Silence warnings for calls on unsupported architectures.
// Unsupported architectures will generate no-ops and test
// will be avoided at runtime anyway.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct GemmTestTraits
    {
        // Size properties of gemm tiles
        enum struct Sizes : uint32_t
        {
            TileAX    = BlocksX * BlockM,
            TileAY    = BlockK,
            TileASize = TileAX * TileAY * sizeof(InputT),

            TileBX    = BlockK,
            TileBY    = BlocksY * BlockN,
            TileBSize = TileBX * TileBY * sizeof(InputT),

            TileCX    = BlocksX * BlockM,
            TileCY    = BlocksY * BlockN,
            TileCSize = TileCX * TileCY * sizeof(ComputeT),

            TileDX    = BlocksX * BlockM,
            TileDY    = BlocksY * BlockN,
            TileDSize = TileDX * TileDY * sizeof(OutputT),
        };

        // Tile costs
        enum struct Cost : uint32_t
        {
            Granularity = 16u,
            DWord       = 4u,

            TileA = ceilDiv((uint32_t)Sizes::TileASize, Granularity* WaveSize* DWord) * Granularity,
            TileB = ceilDiv((uint32_t)Sizes::TileBSize, Granularity* WaveSize* DWord) * Granularity,
            TileC = ceilDiv((uint32_t)Sizes::TileCSize, Granularity* WaveSize* DWord) * Granularity,
            TileD = ceilDiv((uint32_t)Sizes::TileDSize, Granularity* WaveSize* DWord) * Granularity,
        };

        // Architecture we are testing
        enum Arch : bool
        {
            IsWave32 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_32),
            IsWave64 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_64),

            IsGfx908  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX908),
            IsGfx90A  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX90A),
            IsGfx940  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX940),
            IsGfx941  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX941),
            IsGfx942  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX942),
            IsGfx1100 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1100),
            IsGfx1101 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1101),
            IsGfx1102 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1102),

            IsGfx9  = IsGfx908 || IsGfx90A || IsGfx940 || IsGfx941 || IsGfx942,
            IsGfx11 = IsGfx1100 || IsGfx1101 || IsGfx1102,
        };
    };

} // namespace rocWMMA

#endif // ROCWMMA_GEMM_TEST_TRAITS_HPP
