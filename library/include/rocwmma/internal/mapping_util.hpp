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
#ifndef ROCWMMA_MAPPING_UTIL_HPP
#define ROCWMMA_MAPPING_UTIL_HPP

#include <utility>

#include "types.hpp"

namespace rocwmma
{

    // 2D Coordinate
    using Coord2d = std::pair<uint32_t, uint32_t>;

    // Fwd declaration
    struct row_major;
    struct col_major;

    namespace detail
    {
        // TBlockX, TBlockY default to runtime variable query of blockDim.x, blockDim.y
        // if not known at compile time.
        template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
        struct WaveSpace
        {
            using WaveCoordT      = Coord2d;
            using WorkgroupCoordT = Coord2d;
            using WorkgroupDimT   = Coord2d;

            // Current lane normalized to [0, 63].
            __device__ static inline uint32_t localLaneId();

            // Local wave coordinate relative to current workgroup.
            __device__ constexpr static inline WaveCoordT localWaveCoord();

            // Global wave grid coordinate relative to all workgroups.
            __device__ static inline WaveCoordT globalWaveCoord();

            // Global workgroup Id
            __device__ constexpr static inline WorkgroupCoordT workgroupCoord();

            // Size of workgroup, normalized to wave count.
            __device__ constexpr static inline WorkgroupDimT workgroupDim();
        };

        /*
    Matrix coordinate space is analogous to global mfma block space scaled by BlockM and BlockN.
    */
        template <uint32_t BlockHeight, uint32_t BlockWidth>
        struct MatrixSpace
        {
            using MatrixCoordT = Coord2d;
            using BlockCoordT  = Coord2d;

            // Global matrix coordinate space (row, col) transform for a given block grid coordinate.
            __device__ static inline MatrixCoordT fromBlockCoord(BlockCoordT const& blockCoord);
        };

        /*
    Calculate the memory offsets and addresses for a given matrix coordinate or block coordinate.
    */
        template <typename DataOrientation>
        struct DataSpace
        {
            using MatrixCoordT = Coord2d;
            using MatrixSizeT  = Coord2d;

            enum : uint32_t
            {
                MajorIndex = std::is_same<DataOrientation, row_major>::value ? 0 : 1,
                MinorIndex = std::is_same<DataOrientation, row_major>::value ? 1 : 0
            };

            // Determine the leading dimension of a matrix.
            __device__ constexpr static inline auto leadingDim(MatrixSizeT const& matrixSize);

            // Global data coordinate space (1d element) transform for a matrix coordinate.
            __device__ constexpr static inline auto fromMatrixCoord(MatrixCoordT const& matrixCoord,
                                                                    uint32_t            leadingDim);
        };

        template <>
        struct DataSpace<void>;

    } // namespace detail;

    /*
This mapping utility is intended to map from workgroup configuration into
functional wave units. Depending on how the workgroup threads are grouped,
threads are mapped to their work differently, so it is best to assume a
particular configuration.

***Important assumption: ***

This particular mapping assumes that the major dimension for
threadcount is in the blockDim.x element.

This means that blockSize.x threadcounts are multiples of 64,
and blockSize.y threadcounts are in multiples of 1.

blockDim = (waveRows * 64, waveCols).

Wave rows and cols map directly to scaled matrix rows and cols for processing.

***

E.g.
BlockDim of (64, 1) will give a grid of (1, 1) waves, or 1 total wave in the workgroup.
localWaveId of (1, 1) is the wave corresponding to threads ([0 - 63], 1) in the
workgroup.

E.g.
BlockDim of (256, 4) will give a grid of (4, 4) waves, or 16 total waves in the workgroup.
localWaveId of (2, 3) is the wave corresponding to threads ([128 - 191], 3) in the
workgroup.
*/
    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    struct MappingUtil
    {
        using WaveSpace   = detail::WaveSpace<>;
        using MatrixSpace = detail::MatrixSpace<BlockHeight, BlockWidth>;
        using DataSpace   = detail::DataSpace<DataLayout>;

        using WaveCoordT    = typename WaveSpace::WaveCoordT;
        using BlockCoordT   = typename MatrixSpace::BlockCoordT;
        using MatrixCoordT  = typename MatrixSpace::MatrixCoordT;
        using WorkgroupDimT = typename WaveSpace::WorkgroupDimT;

        /// Current wave perspective

        // Current lane of current wave
        __device__ static inline uint32_t laneId();

        // Local wave coordinate relative to workgroup
        __device__ static inline WaveCoordT waveCoord();

        // Global block (grid) coordinate of current wave
        __device__ static inline BlockCoordT blockCoord();

        // Matrix coordinate of current wave
        __device__ static inline MatrixCoordT matrixCoord();

        // Data address of current wave
        __device__ static inline DataT const* dataCoord(DataT const* baseAddr, uint32_t ldm);
        __device__ static inline DataT*       dataCoord(DataT* baseAddr, uint32_t ldm);

        /// Current workgroup perspective

        __device__ static inline WorkgroupDimT workgroupDim();

        /// Coordinate override helpers

        // Current global wave coordinate with row override
        __device__ static inline BlockCoordT blockCoordM(uint32_t m);

        // Current global wave coordinate with col override
        __device__ static inline BlockCoordT blockCoordN(uint32_t n);

        // Matrix coordinate of current wave with row override
        __device__ static inline MatrixCoordT matrixCoordM(uint32_t m);

        // Matrix coordinate of current wave with col override
        __device__ static inline MatrixCoordT matrixCoordN(uint32_t n);

        /// Conversion helpers

        // Convert from any block coord to matrix coord
        __device__ static inline MatrixCoordT matrixCoord(BlockCoordT const& blockCoord);

        // Convert from any matrix coord to data offset
        __device__ static inline uint32_t dataOffset(MatrixCoordT const& matrixCoord, uint32_t ldm);

        // Convert from any matrix coord to data address
        __device__ static inline DataT const*
            dataCoord(DataT const* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm);
        __device__ static inline DataT*
            dataCoord(DataT* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm);
    };

} // namespace rocwmma

#include "mapping_util_impl.hpp"

#endif // ROCWMMA_MAPPING_UTIL_HPP
