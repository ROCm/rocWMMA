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

template <typename Layout = common>
struct MatrixUtil
{
    /*
    Calculate the true grid Id for each thread, taking into account
    that there could be multiple waves per thread block (e.g. blockDim / WAVE_SIZE > 1).
    Each wave will calculate one WMMA block.
    */
    __device__ static inline auto mapWaveToWMMAGrid()
        -> std::pair<uint32_t, uint32_t> // BlockM, BlockN
    {
        return std::make_pair((blockIdx.y * blockDim.y + threadIdx.y), // ROW
                              (blockIdx.x * blockDim.x + threadIdx.x) / AMDGCN_WAVE_SIZE); // COL
    }
};

template <>
struct MatrixUtil<row_major> : public MatrixUtil<common>
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

    /*
    For each wave, calculate the data address of its WMMA block from a global pointer.
    */
    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    __device__ static inline DataT* mapWaveToWMMABlock(DataT const* addr, uint32_t ldm)
    {
        // Unpack the true grid id
        auto gridIdx = mapWaveToWMMAGrid();

        // Align pointer to data starting at (row, col)
        return const_cast<float*>(addr) + (std::get<0>(gridIdx) * BlockM * ldm) + // from row
               (std::get<1>(gridIdx) * BlockN); // from col
    }
};

template <>
struct MatrixUtil<col_major> : public MatrixUtil<common>
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

    /*
    For each wave, calculate the data address of its WMMA block from a global pointer.
    */
    template <typename DataT, uint32_t BlockM, uint32_t BlockN>
    __device__ static inline DataT* mapWaveToWMMABlock(DataT const* addr, uint32_t ldm)
    {
        // Unpack the true grid id
        auto gridIdx = mapWaveToWMMAGrid();

        // Align pointer to data starting at (row, col)
        return const_cast<float*>(addr) + (std::get<0>(gridIdx) * BlockM) + // from row
               (std::get<1>(gridIdx) * BlockN * ldm); // from col
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
