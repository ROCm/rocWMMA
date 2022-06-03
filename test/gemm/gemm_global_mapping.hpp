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
    namespace GlobalMapping
    {
        namespace detail
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
            struct MappingBase
            {
                /*
                * This global mapping aligns each wave with a gemm C/D tile
                * that the wave will compute independently. All calculations
                * are in matrix coordinates (Coord2d).
                *
                * Coordinates are scaled into three different levels:
                * - Macro level: context of work for entire workgroup
                * - Wave level: context of work for each wave
                * - Block level: context of work for each mfma block
                *
                * Waves computing tiles on the same row (matrix A) and waves
                * computing tiles on the same col (matrix B) have opportunities
                * for data sharing. As a result, waves collaboration is limited
                * to tiles in same locality.
                *
                * Block size:      (BlockM x BlockN)
                * Wave tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
                * Macro Tile size: (WgDim.x * WaveTileSize.x) x (WgDim.y * WaveTileSize.y)
                *
                * Wave data share A: same row
                * Wave data share B: same col
                *
                * Global C layout & workgroup assignment for WGDim = (2x2):
                *
                * W (X, Y) = wave row X, col Y
                *
                *
                *                                     |--------- Macro Tile Y-------------|
                *                                     |-- Wave Tile Y --|
                *                                     |-BlockN-|
                *
                *                                      BlockN x BlocksY   BlockN x BlocksY
                *                                     |<--------------->|<--------------->|
                *      _ _   _ _      _ _          ___  ________ ________ ________ ________
                *       |     |        |            ^  |        |        |        |        |
                *       | Wave| BlockM |   BlockM   |  |        W        |        W        |
                *       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
                *       |  X  |            BlocksX  |  |                 |                 |
                * Macro |     |                     |  |                 |                 |
                *  Tile |    _|_                   _v_ |________|________|________|________|
                *   X   |                           ^  |        |        |        |        |
                *       |                  BlockM   |  |        W        |        W        |
                *       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
                *       |                  BlocksX  |  |                 |                 |
                *       |                           |  |                 |                 |
                *      _|_                         _v_ |________|________|________|________|
                *
                *
                * TLDR: Global mapping aligns global offsets for A / B / C / D  MFMA blocks.
                * Configures fragment and buffer types for per-wave responsibility of
                * BlocksX * BlocksY MFMA blocks.
                */

                // Block fragments that are used as inputs to MFMA operations.
                using MfmaFragA   = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
                using MfmaFragB   = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
                using MfmaFragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutC>;
                using MfmaFragD   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutD>;
                using MfmaFragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

                // Mfma fragment buffers that are compatible with the gemm driver
                using MfmaBuffA   = MfmaFragA[BlocksX];
                using MfmaBuffB   = MfmaFragB[BlocksY];
                using MfmaBuffC   = MfmaFragC[BlocksX][BlocksY];
                using MfmaBuffD   = MfmaFragD[BlocksX][BlocksY];
                using MfmaBuffAcc = MfmaFragAcc[BlocksX][BlocksY];

                using WaveSpace = typename rocwmma::detail::WaveSpace;

                // Projection of C coordinate in direction of A
                template <typename CoordC>
                __device__ constexpr static inline auto projCoordA(CoordC const& coordC);

                // Projection of C coordinate in direction of B
                template <typename CoordC>
                __device__ constexpr static inline auto projCoordB(CoordC const& coordC);

                // Tile sizes:
                // Macro tile = tile processed by entire workgroup
                // Wave tile = tile processed by current wave
                // Block size = mfma block size
                __device__ constexpr static inline auto macroTileSizeC();
                __device__ constexpr static inline auto waveTileSizeC();
                __device__ constexpr static inline auto blockSizeC();
                __device__ constexpr static inline auto kDim();

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

        } // namespace detail

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
        struct BlockLevelMapping : public detail::MappingBase<BlockM,
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
                                                              BlocksY>
        {
            /*
            * This flavour of Global Mapping targets individual waves iteratively
            * to their MFMA blocks in a grid:
            * BlocksX x (BlockM x BlockK) for A and
            * BlocksY x (BlockN x BlockK) for B.
            *
            * This leads to favourable MFMA layouts directly from Global Reads,
            * but may not be the most efficient in some layouts.
            */

            using Base = detail::MappingBase<BlockM,
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

            using GRFragA = typename Base::MfmaFragA;
            using GRFragB = typename Base::MfmaFragB;

            using GRBuffA = typename Base::MfmaBuffA; // Multiple fragments
            using GRBuffB = typename Base::MfmaBuffB; // Multiple fragments
        };

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
        struct WaveLevelMapping : public detail::MappingBase<BlockM,
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
                                                             BlocksY>
        {
            /*
            * This flavour of Global Mapping targets individual waves at one combined
            * Global Read fragment sized:
            * (BlocksX * BlockM) x BlockK for A and
            * (BlocksY * BlockN) x BlockK for B.
            *
            * This global read fragment is not MFMA friendly, however when written to LDS
            * smaller MFMA friendly fragment may be read directly from the same LDS layout.
            * Larger GR may be more efficient in certain layouts for data pipelining.
            */
            using Base = detail::MappingBase<BlockM,
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

            using GRFragA = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, InputT, LayoutA>;
            using GRFragB = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, InputT, LayoutB>;

            using GRBuffA = GRFragA; // Single fragment
            using GRBuffB = GRFragB; // Single fragment
        };

    } // namespace GlobalMapping

} // namespace rocwmma

#include "gemm_global_mapping_impl.hpp"

#endif // GEMM_GLOBAL_MAPPING_HPP
