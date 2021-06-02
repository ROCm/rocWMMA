#ifndef WMMA_MAPPING_UTIL_H
#define WMMA_MAPPING_UTIL_H

#include <hip/hip_runtime.h>

#include <utility>

namespace _MappingUtil
{
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

    // Workgroup configuration: These three functions define how block dimensions are handled.

    __device__ static inline uint32_t laneId();

    // Convert from thread count to wave count
    __device__ constexpr static inline auto
        waveCount(std::pair<uint32_t, uint32_t> const& threadCount)
            -> std::pair<uint32_t, uint32_t>;

    // Convert from wave count to thread count
    __device__ constexpr static inline auto
        threadCount(std::pair<uint32_t, uint32_t> const& waveCount)
            -> std::pair<uint32_t, uint32_t>;

    // Coordinate spaces

    struct WaveSpace
    {
        using CoordT = std::pair<uint32_t, uint32_t>;

        // Size of workgroup, normalized to wave count.
        __device__ static inline auto workgroupDim() -> CoordT;

        // Global workgroup Id
        __device__ static inline auto workgroupCoord() -> CoordT;

        // Grid coordinate of the current wave local to the workgroup.
        __device__ static inline auto localWaveCoord() -> CoordT;

        // Grid coordinate of the current wave in global context of all waves from
        // all workgroups.
        __device__ static inline auto globalWaveCoord() -> CoordT;

        // Each wave has 64 lanes normalized to [0, 63] to do work.
        __device__ static inline uint32_t localLaneId();
    };

    /*
    Calculate the matrix coordinate space (row, col) for a given block coordinate.
    Matrix coordinate space is analogous to global block space scaled by BlockM and BlockN.
    */
    template <uint32_t BlockM, uint32_t BlockN>
    struct MatrixSpace
    {
        using CoordT = std::pair<uint32_t, uint32_t>;

        __device__ static inline auto fromBlockCoord(CoordT const& blockCoord) -> CoordT;
    };

    /*
    Calculate the memory address for a given matrix coordinate or block coordinate.
    */
    template <typename DataT, uint32_t BlockM, uint32_t BlockN, typename DataLayout>
    struct DataSpace
    {
        using CoordT = std::pair<uint32_t, uint32_t>;

        __device__ static inline DataT*
            fromMatrixCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& matrixCoord);
        __device__ static inline DataT*
            fromBlockCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& blockCoord);
    };

} // namespace _MappingUtil

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
struct MappingUtil
{
    using WaveSpace   = _MappingUtil::WaveSpace;
    using MatrixSpace = _MappingUtil::MatrixSpace<BlockM, BlockN>;
    using DataSpace   = _MappingUtil::DataSpace<DataT, BlockM, BlockN, DataLayout>;
    using CoordT      = std::pair<uint32_t, uint32_t>;

    /// Current wave perspective

    // Current lane of current wave
    __device__ static inline uint32_t laneId();

    // Current local wave coordinate
    __device__ static inline auto waveCoord() -> CoordT;

    // Current global wave coordinate
    __device__ static inline auto blockCoord() -> CoordT;

    // Matrix coordinate of current wave
    __device__ static inline auto matrixCoord() -> CoordT;

    // Data address of current wave
    __device__ static inline DataT* dataCoord(DataT const* baseAddr, uint32_t ldm);

    /// Current workgroup perspective

    __device__ static inline auto workgroupDim() -> CoordT;

    /// Coordinate override helpers

    // Current global wave coordinate with row override
    __device__ static inline auto blockCoordM(uint32_t m) -> CoordT;

    // Current global wave coordinate with col override
    __device__ static inline auto blockCoordN(uint32_t n) -> CoordT;

    // Matrix coordinate of current wave with row override
    __device__ static inline auto matrixCoordM(uint32_t m) -> CoordT;

    // Matrix coordinate of current wave with col override
    __device__ static inline auto matrixCoordN(uint32_t n) -> CoordT;

    /// Conversion helpers

    // Convert from any block coord to matrix coord
    __device__ static inline auto matrixCoord(CoordT const& blockCoord) -> CoordT;

    // Convert from any matrix coord to data address
    __device__ static inline DataT*
        dataCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& matrixCoord);
};

// template <size_t WaveRows,
//           size_t WaveCols,
//           size_t BlockM,
//           size_t BlockN,
//           size_t BlockK>
// struct LaunchUtil
// {
//     static inline auto gridDim(uint32_t M, uint32_t N, uint32_t K) -> dim3
//     {
//         return dim3(ceilDiv(M, BlockM * WaveRows), ceilDiv(N, BlockM * WaveCols));
//     }

//     static inline auto blockDim() -> dim3
//     {
//         auto threads = _MappingUtil::threadCount(std::make_pair<uint32_t, uint32_t>(WaveRows, WaveCols));
//         return dim3(std::get<0>(threads), std::get<1>(threads));
//     }
// };

#include "MappingUtil_impl.h"

#endif // WMMA_MAPPING_UTIL_H
