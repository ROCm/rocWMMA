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
#ifndef GEMM_GLOBAL_MAPPING_HPP
#define GEMM_GLOBAL_MAPPING_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace CooperativeGemm
    {
        template<uint32_t BlockM,
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
        struct GlobalMapping
        {
            ///
            // This global mapping aligns each wave with a gemm C/D Tile
            // that the wave will compute independently. All calculations
            // are in matrix coordinates (Coord2d).
            //
            // Waves computing tiles on the same row (matrix A) and waves
            // computing tiles on the same col (matrix B) have opportunities
            // for data sharing. As a result, waves collaboration is limited
            // to tiles in same locality.
            // 
            // Wave tile size:     (BlocksX * BlockM) x (BlocksY * BlockN)
            // WG MacroTile size: (WgDim.x * BlocksX * BlockM) x (WgDim.y * BlocksY * BlockN)
            //
            // Wave data share A: same row
            // Wave data share B: same col
            //
            // Global C layout & workgroup assignment:
            //
            //                                 MacroTile Y
            //                      |<----------------------------------| 
            //          
            //                       BlocksY x BlockN   BlocksY x BlockN
            //                      |<--------------->|<--------------->|
            //      __            __  ________ ________ ________ ________
            //        ^            ^ |        |        |        |        |
            //        |   BlocksX  | |        W        |        W        |
            //        |      x     | |__   (0, 0)    __|__   (0, 1)    __| 
            //        |   BlockM   | |                 |                 |
            //  Macro |            | |                 |                 |
            //   Tile |           _v |________|________|________|________|
            //    X   |            ^ |        |        |        |        |
            //        |   BlocksX  | |        W        |        W        |
            //        |      x     | |__   (1, 0)    __|__   (1, 1)    __| 
            //        |   BlockM   | |                 |                 |
            //        |            | |                 |                 |
            //       _v           _v |________|________|________|________|
            
            // Global wave tile fragments that are loaded collaboratively
            using GRFragA = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, InputT, LayoutA>;
            using GRFragB = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, InputT, LayoutB>;

            // Block fragments that are used as inputs to MFMA operations.
            using MfmaFragA = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
            using MfmaFragB = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
            using MfmaFragC = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutC>;
            using MfmaFragD = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutD>;
            using MfmaFragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

            // Helper to access fragment dimensions                       
            template<typename FragT>
            using IOShape = typename FragT::IOConfig::IOShape;

            // Projection of C coordinate in direction of A
            template<typename CoordC>
            __device__ constexpr static inline auto projCoordA(CoordC const& coordC);

            // Projection of C coordinate in direction of B
            template<typename CoordC>
            __device__ constexpr static inline auto projCoordB(CoordC const& coordC);

            // Full macro tile size
            __device__ constexpr static inline auto macroTileSizeC();

            // Global matrix coordinate of macro tile for the current workgroup
            __device__ constexpr static inline auto macroTileCoordC();

            // The offset from macro tile for current wave
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();
            __device__ constexpr static inline auto waveOffsetC();

            // The local offsets between mfma blocks
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();
            __device__ constexpr static inline auto blockOffsetC();

            // The base global matrix coordinate of the current wave tile.
            // MacroTile coord + wave offset
            __device__ constexpr static inline auto matrixCoordA();
            __device__ constexpr static inline auto matrixCoordB();
            __device__ constexpr static inline auto matrixCoordC();

            // The matrix offset to the next step in the k dimension
            __device__ constexpr static inline auto kStepA();
            __device__ constexpr static inline auto kStepB();
        };

    } // namespace CooperativeGemm

} // namespace rocwmma

#include "gemm_global_mapping_impl.hpp"

#endif // GEMM_GLOBAL_MAPPING_HPP