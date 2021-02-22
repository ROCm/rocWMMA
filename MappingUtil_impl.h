#ifndef WMMA_MAPPING_UTIL_IMPL_H
#define WMMA_MAPPING_UTIL_IMPL_H

#include <hip/hip_runtime.h>
#include <utility>


#include "Constants.h"
#include "MappingUtil.h"
#include "Types.h"


namespace _MappingUtil
{
    // Helper funcs

    // Workgroup configuration
    // According to our assumption, the major thread dimension is X, so
    // this determines our laneId.
    __device__ inline uint32_t laneId()
    {
        return threadIdx.x / AMDGCN_WAVE_SIZE;
    }

    __device__ constexpr inline auto waveCount(std::pair<uint32_t, uint32_t> const& threadCount) -> std::pair<uint32_t, uint32_t>
    {
        return std::make_pair(
            std::get<0>(threadCount) / AMDGCN_WAVE_SIZE, // ROW
            std::get<1>(threadCount));                   // COL
    }

    __device__ constexpr inline auto threadCount(std::pair<uint32_t, uint32_t> const& waveCount) -> std::pair<uint32_t, uint32_t>
    {
        return std::make_pair(
            std::get<0>(waveCount) * AMDGCN_WAVE_SIZE, // ROW
            std::get<1>(waveCount));                   // COL
    }

    // struct WaveSpace

    __device__ inline auto WaveSpace::workgroupDim() -> CoordT
    {
        return waveCount(std::make_pair(blockDim.x, blockDim.y));
    }

    __device__ inline auto WaveSpace::workgroupCoord() -> CoordT
    {
        return std::make_pair(blockIdx.x, blockIdx.y);
    }

    __device__ inline auto WaveSpace::localWaveCoord() -> CoordT
    {
        return waveCount(std::make_pair(threadIdx.x, threadIdx.y));
    }

    __device__ inline auto WaveSpace::globalWaveCoord() -> CoordT
    {
        return waveCount(std::make_pair(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y));
    }

    __device__ inline uint32_t WaveSpace::localLaneId()
    {
        return laneId();
    }

    // template <uint32_t BlockM, uint32_t BlockN>
    // struct MatrixSpace

    template <uint32_t BlockM, uint32_t BlockN>
    __device__ inline auto MatrixSpace<BlockM, BlockN>::fromBlockCoord(CoordT const& blockCoord) -> CoordT
    {
        // Map block to matrix coordinate space.
        return std::make_pair(std::get<0>(blockCoord) * BlockM,  // ROW
                              std::get<1>(blockCoord) * BlockN); // COL
    }

    // template <typename DataT, uint32_t BlockM, uint32_t BlockN, typename DataLayout>
    // struct DataSpace;

    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    struct DataSpace<DataT, BlockM, BlockN, row_major>
    {
        using CoordT = std::pair<uint32_t, uint32_t>;

        __device__ static inline DataT* fromMatrixCoord(DataT const* addr, uint32_t ldm, CoordT const& matrixCoord)
        {
            // Align pointer to data starting at (row, col)
            return const_cast<float*>(addr) + 
                   std::get<0>(matrixCoord) * ldm + // Row
                   std::get<1>(matrixCoord);        // Col
        }

        __device__ static inline DataT* fromBlockCoord(DataT const* addr, uint32_t ldm, CoordT const& blockCoord)
        {
            // First map from matrix then fwd.
            return fromMatrixCoord(MatrixSpace<BlockM, BlockN>::fromBlockCoord(blockCoord));
        }
    };

    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    struct DataSpace<DataT, BlockM, BlockN, col_major>
    {
        using CoordT = std::pair<uint32_t, uint32_t>;

        __device__ static inline DataT* fromMatrixCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& matrixCoord)
        {
            // Align pointer to data starting at (row, col)
            return const_cast<float*>(baseAddr) +
                   std::get<0>(matrixCoord) +      // Row
                   std::get<1>(matrixCoord) * ldm; // Col
        }

        __device__ static inline DataT* fromBlockCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& blockCoord)
        {
            // First map from matrix then fwd.
            return fromMatrixCoord(baseAddr, ldm, MatrixSpace<BlockM, BlockN>::fromBlockCoord(blockCoord));
        }
    };

} // namespace _MappingUtil

// template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
// struct MappingUtil

/// Current wave perspective

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline uint32_t MappingUtil<BlockM, BlockN, DataT, DataLayout>::laneId()
{
    return WaveSpace::localLaneId();
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::waveCoord() -> CoordT
{
    return WaveSpace::localWaveCoord();
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::blockCoord() -> CoordT
{
    return WaveSpace::globalWaveCoord();
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::matrixCoord() -> CoordT
{
    return MatrixSpace::fromBlockCoord(blockCoord());
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline DataT* MappingUtil<BlockM, BlockN, DataT, DataLayout>::dataCoord(DataT const* baseAddr, uint32_t ldm)
{
    return DataSpace::fromMatrixCoord(baseAddr, ldm, matrixCoord());
}

/// Coordinate override helpers

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::blockCoordM(uint32_t m) -> CoordT
{
    auto coord = blockCoord();
    std::get<0>(coord) = m;
    return coord;
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::blockCoordN(uint32_t n) -> CoordT
{
    auto coord = blockCoord();
    std::get<1>(coord) = n;
    return coord;
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::matrixCoordM(uint32_t m) -> CoordT
{
    auto coord = matrixCoord();
    std::get<0>(coord) = m;
    return coord;
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::matrixCoordN(uint32_t n) -> CoordT
{
    auto coord = matrixCoord();
    std::get<1>(coord) = n;
    return coord;
}

/// Conversion helpers

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline auto MappingUtil<BlockM, BlockN, DataT, DataLayout>::matrixCoord(CoordT const& blockCoord) -> CoordT
{
    return MatrixSpace::fromBlockCoord(blockCoord);
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
__device__ inline DataT* MappingUtil<BlockM, BlockN, DataT, DataLayout>::dataCoord(DataT const* baseAddr, uint32_t ldm, CoordT const& matrixCoord)
{
    return DataSpace::fromMatrixCoord(baseAddr, ldm, matrixCoord);
}

#endif // WMMA_MAPPING_UTIL_IMPL_H
