#ifndef WMMA_UTILS_H
#define WMMA_UTILS_H

#include <assert.h>
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
    Grid coordinate space is analogous to two dimensional block ID.
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
    Matrix coordinate space is analogous to grid space scaled by BlockM and BlockN.
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

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
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
    __device__ static inline auto gridCoord() -> decltype(GridMapping::gridCoord())
    {
        return GridMapping::gridCoord();
    }

    // 2d block coord of current wave.
    __device__ static inline auto blockCoord() -> decltype(BlockMapping::fromGrid())
    {
        return BlockMapping::fromGrid();
    }

    // 1d data coord of current wave.
    __device__ static inline auto dataCoord(DataT const* addr, uint32_t ldm)
        -> decltype(DataMapping::fromBlock(addr, ldm))
    {
        return DataMapping::fromBlock(addr, ldm);
    }

    /// Helpers to navigate blocks in the current column.

    // Get current grid coordinate and override M
    __device__ static inline auto gridCoordM(uint32_t m) -> decltype(GridMapping::gridCoord())
    {
        auto gridCoordM         = gridCoord();
        std::get<0>(gridCoordM) = m;
        return gridCoordM;
    }

    // Get current 2D block coordinate and override grid coord M
    __device__ static inline auto blockCoordM(uint32_t gridM)
        -> decltype(BlockMapping::fromGrid(gridCoordM(gridM)))
    {
        return BlockMapping::fromGrid(gridCoordM(gridM));
    }

    // Get 1D data coordinate and override grid coord with M
    __device__ static inline auto dataCoordM(DataT const* addr, uint32_t ldm, uint32_t gridM)
        -> decltype(DataMapping::fromBlock(addr, ldm, blockCoordM(gridM)))
    {
        return DataMapping::fromBlock(addr, ldm, blockCoordM(gridM));
    }

    /// Helpers to navigate blocks in the current row.

    // Get current grid coordinate and override N
    __device__ static inline auto gridCoordN(uint32_t n) -> decltype(GridMapping::gridCoord())
    {
        auto gridCoordN         = gridCoord();
        std::get<1>(gridCoordN) = n;
        return gridCoordN;
    }

    // Get current 2D block coordinate and override grid coord N
    __device__ static inline auto blockCoordN(uint32_t gridN)
        -> decltype(BlockMapping::fromGrid(gridCoordN(gridN)))
    {
        return BlockMapping::fromGrid(gridCoordN(gridN));
    }

    // Get 1D data coordinate and override grid coord with M
    __device__ static inline auto dataCoordN(DataT const* addr, uint32_t ldm, uint32_t gridN)
        -> decltype(DataMapping::fromBlock(addr, ldm, blockCoordN(gridN)))
    {
        return DataMapping::fromBlock(addr, ldm, blockCoordN(gridN));
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
        for(int i = 0; i < m; ++i)
        {
            std::cout << "[ ";
            for(int j = 0; j < n; ++j)
            {
                // (Row, col)
                std::cout << mat[i * n + j] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                // Count up in ascending order, alternating evens and odds
                // with respective positive / negative
                int32_t val = i * n + j;
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
        for(int i = 0; i < m; ++i)
        {
            std::cout << "[ ";
            for(int j = 0; j < n; ++j)
            {
                // (Row, col)
                std::cout << mat[i + j * m] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                // Count up in ascending order, alternating evens and odds
                // with respective positive / negative
                int32_t val    = i * n + j;
                mat[i + j * m] = val % 2 ? -val : val;
            }
        }
    }
};

template <typename T,
          size_t BlockM  = 0,
          size_t BlockN  = 0,
          size_t BlockK  = 0,
          size_t TBlockX = 0, // Launch param thread block size
          size_t TBlockY = 0> // Launch param thread block size
struct BlockGeometry
{
    static constexpr size_t threads_per_wave  = 64;
    static constexpr size_t elements_per_vgpr = 256 / sizeof(T);

    static constexpr auto blockStrides() -> std::tuple<size_t, size_t, size_t>
    {
        return std::tuple<size_t, size_t, size_t>(BlockM, BlockN, BlockK);
    }

    static constexpr auto blockLaunchDims() -> std::tuple<size_t, size_t>
    {
        return std::tuple<size_t, size_t>(TBlockX, TBlockY);
    }

    // How many mxnxk blocks total.
    static constexpr auto gridDim(size_t M, size_t N, size_t K)
        -> std::tuple<size_t, size_t, size_t>
    {
        return std::tuple<size_t, size_t, size_t>(ceilDiv(M, BlockM * TBlockX / threads_per_wave),
                                                  ceilDiv(N, BlockN * TBlockY),
                                                  ceilDiv(K, BlockK));
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

template <typename LayoutA, typename LayoutB, typename LayoutC, typename InputT, typename ComputeT>
void gemmCPU(std::vector<InputT> const& a,
             std::vector<InputT> const& b,
             std::vector<ComputeT>&     c,
             int                        M,
             int                        N,
             int                        K,
             ComputeT                   alpha,
             ComputeT                   beta)
{
    int lda = std::is_same<LayoutA, row_major>::value ? K : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
    int ldc = std::is_same<LayoutC, row_major>::value ? N : M;

    for(int i = 0; i < M; ++i) // Row
    {
        for(int j = 0; j < N; ++j) // Col
        {
            float accum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                auto indexA
                    = std::is_same<LayoutA, row_major>::value ? (i * lda + k) : (i + lda * k);
                auto indexB
                    = std::is_same<LayoutB, row_major>::value ? (k * ldb + j) : (k + j * ldb);
                accum += a[indexA] * b[indexB];
            }

            auto indexC = std::is_same<LayoutC, row_major>::value ? (i * ldc + j) : (i + j * ldc);
            c[indexC]   = alpha * accum + beta * c[indexC];
        }
    }
}

template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
void compareEqual(std::vector<TypeA> const& a, std::vector<TypeB> const& b, int M, int N)
{
    assert(a.size() == b.size() && "A and B are not the same size");
    assert(a.size() == M * N && "A and B do not match size M x N");
    int lda = std::is_same<LayoutA, row_major>::value ? N : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : M;

    for(int i = 0; i < M; ++i) // Row
    {
        for(int j = 0; j < N; ++j) // Col
        {
            auto indexA = std::is_same<LayoutA, row_major>::value ? (i * lda + j) : (i + j * lda);
            auto indexB = std::is_same<LayoutB, row_major>::value ? (i * ldb + j) : (i + j * ldb);

            if(a[indexA] != b[indexB])
            {
                std::cout << "No match: Element( " << i << ", " << j << " )\n";
                std::cout << "(A, B): ( " << a[indexA] << ", " << b[indexB] << ")\n";
            }
        }
    }
}

#endif // WMMA_UTILS_H
