#ifndef WMMA_UTILS_H
#define WMMA_UTILS_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include "Constants.h"
#include "Types.h"

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

namespace _MappingUtil
{
    /*
    Calculate the WMMA block origin in grid coordinate space (row, col).
    Grid coordinate space is analagous to two dimensional block ID.
    Each thread in the wave is assigned this ID.
    */
    struct GridMapping
    {
        enum : uint32_t
        {
            WAVE_SIZE = AMDGCN_WAVE_SIZE
        };

        __device__ static inline auto gridCoord() -> std::pair<uint32_t, uint32_t> // BlockM, BlockN
        {
            return std::make_pair((blockIdx.y * blockDim.y + threadIdx.y), // ROW
                                  (blockIdx.x * blockDim.x + threadIdx.x)
                                      / AMDGCN_WAVE_SIZE); // COL
        }
    };

    /*
    Calculate the WMMA block origin in matrix coordinate space (row, col).
    Matrix coordinate space is analagous to grid space scaled by BlockM and BlockN.
    Each thread in the wave is assigned to this coordinate.
    */
    template <uint32_t BlockM, uint32_t BlockN>
    struct BlockMapping
    {
        __device__ static inline auto fromGrid(std::pair<uint32_t, uint32_t> const& gridCoord
                                               = GridMapping::gridCoord())
            -> std::pair<uint32_t, uint32_t> // BlockM, BlockN
        {
            // Map block to matrix coordinate space.
            return std::make_pair(std::get<0>(gridCoord) * BlockM, // ROW
                                  std::get<1>(gridCoord) * BlockN); // COL
        }
    };

    /*
    Calculate the WMMA block origin in 1D address coordinate (index).
    Each thread in the wave is assigned to this data index.
    */
    template <typename DataT, uint32_t BlockM, uint32_t BlockN, typename Layout>
    struct DataMapping;

    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    struct DataMapping<DataT, BlockM, BlockN, row_major>
    {
        __device__ static inline DataT* fromBlock(DataT const*                         addr,
                                                  uint32_t                             ldm,
                                                  std::pair<uint32_t, uint32_t> const& blockCoord
                                                  = BlockMapping<BlockM, BlockN>::fromGrid())
        {
            // Align pointer to data starting at (row, col)
            return const_cast<float*>(addr) + std::get<0>(blockCoord) * ldm + // Row
                   std::get<1>(blockCoord); // Col
        }

        __device__ static inline DataT* fromGrid(DataT const*                         addr,
                                                 uint32_t                             ldm,
                                                 std::pair<uint32_t, uint32_t> const& gridCoord
                                                 = GridMapping::gridCoord())
        {
            // First map from grid then fwd.
            return fromBlock(BlockMapping<BlockM, BlockN>::fromGrid(gridCoord));
        }
    };

    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    struct DataMapping<DataT, BlockM, BlockN, col_major>
    {
        __device__ static inline DataT* fromBlock(DataT const*                         addr,
                                                  uint32_t                             ldm,
                                                  std::pair<uint32_t, uint32_t> const& blockCoord
                                                  = BlockMapping<BlockM, BlockN>::fromGrid())
        {
            // Align pointer to data starting at (row, col)
            return const_cast<float*>(addr) + std::get<0>(blockCoord) + // Row
                   std::get<1>(blockCoord) * ldm; // Col
        }

        __device__ static inline DataT* fromGrid(DataT const*                         addr,
                                                 uint32_t                             ldm,
                                                 std::pair<uint32_t, uint32_t> const& gridCoord
                                                 = GridMapping::gridCoord())
        {
            // Retrieve block coordinates
            return fromBlock(BlockMapping<BlockM, BlockN>::fromGrid(gridCoord));
        }
    };
}

template <typename DataT, uint32_t BlockM, uint32_t BlockN, typename Layout>
struct MappingUtil
{
    using GridMapping  = _MappingUtil::GridMapping;
    using BlockMapping = _MappingUtil::BlockMapping<BlockM, BlockN>;
    using DataMapping  = _MappingUtil::DataMapping<DataT, BlockM, BlockN, Layout>;

    enum : uint32_t
    {
        WAVE_SIZE = GridMapping::WAVE_SIZE
    };

    // 2d grid Coord of current wave.
    // All threads
    __device__ static inline auto gridCoord() -> decltype(GridMapping::gridCoord())
    {
        return GridMapping::gridCoord();
    }

    // 2d block coord of current wave.
    // All threads
    __device__ static inline auto blockCoord() -> decltype(BlockMapping::fromGrid())
    {
        return BlockMapping::fromGrid();
    }

    // 1d data coord of current wave.
    // All threads
    __device__ static inline auto dataCoord(DataT const* addr, uint32_t ldm)
        -> decltype(DataMapping::fromBlock(addr, ldm))
    {
        return DataMapping::fromBlock(addr, ldm);
    }

    // Get 2D grid coordinate and set row to 0
    __device__ static inline auto gridCoordM0() -> decltype(GridMapping::gridCoord())
    {
        auto gridCoordM0         = gridCoord();
        std::get<0>(gridCoordM0) = 0;
        return gridCoordM0;
    }

    // Get 2D grid coordinate and set col to 0
    __device__ static inline auto gridCoordN0() -> decltype(GridMapping::gridCoord())
    {
        auto gridCoordN0         = gridCoord();
        std::get<1>(gridCoordN0) = 0;
        return gridCoordN0;
    }

    // Get 2D block coordinate and set row to 0
    __device__ static inline auto blockCoordM0() -> decltype(BlockMapping::fromGrid(gridCoordM0()))
    {
        return BlockMapping::fromGrid(gridCoordM0());
    }

    // Get 2D block coordinate and set col to 0
    __device__ static inline auto blockCoordN0() -> decltype(BlockMapping::fromGrid(gridCoordN0()))
    {
        return BlockMapping::fromGrid(gridCoordN0());
    }

    // Get 1D data coordinate and set row to 0
    __device__ static inline auto dataCoordM0(DataT const* addr, uint32_t ldm)
        -> decltype(DataMapping::fromBlock(addr, ldm, blockCoordM0()))
    {
        return DataMapping::fromBlock(addr, ldm, blockCoordM0());
    }

    // Get 1D data coordinate and set col to 0
    __device__ static inline auto dataCoordN0(DataT const* addr, uint32_t ldm)
        -> decltype(DataMapping::fromBlock(addr, ldm, blockCoordN0()))
    {
        return DataMapping::fromBlock(addr, ldm, blockCoordN0());
    }
};

template <typename Layout>
struct MatrixUtil;

template <>
struct MatrixUtil<row_major>
{
    template <typename DataT>
    __host__ static inline void print(std::vector<DataT> const& mat, uint32_t m, uint32_t n)
    {
        for(int i = 0; i < n; ++i)
        {
            std::cout << "[ ";
            for(int j = 0; j < m; ++j)
            {
                // (Row, col)
                std::cout << mat[i * m + j] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                // Count up in ascending order, alternating evens and odds
                // with respective positive / negative
                int32_t val = i * m + j;
                mat[val]    = val % 2 ? -val : val;
            }
        }
    }
};

template <>
struct MatrixUtil<col_major>
{
    template <typename DataT>
    __host__ static inline void print(std::vector<DataT> const& mat, uint32_t m, uint32_t n)
    {
        for(int i = 0; i < n; ++i)
        {
            std::cout << "[ ";
            for(int j = 0; j < m; ++j)
            {
                // (Row, col)
                std::cout << mat[j * n + i] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                // Count up in ascending order, alternating evens and odds
                // with respective positive / negative
                int32_t val    = i * m + j;
                mat[j * n + i] = val % 2 ? -val : val;
            }
        }
    }
};

template <int M, int N, int K>
void validateC(std::vector<float> const& a,
               std::vector<float> const& b,
               std::vector<float> const& c)
{
    for(int i = 0; i < M; i++)
    {
        auto rowStartA = i * K;
        for(int j = 0; j < N; j++)
        {
            auto  colStartB = j;
            float result    = 0.0f;
            for(int k = 0; k < K; k++)
            {
                result += (a[rowStartA + k] * b[colStartB + k * N]);
            }

            if(c[i * M + j] != result)
            {
                std::cout << "No match: C( " << i << ", " << j << " )\n";
                std::cout << "(Expected, actual): ( " << result << ", " << c[i * M + j] << ")\n";
            }
        }
    }
}

#endif // WMMA_UTILS_H
