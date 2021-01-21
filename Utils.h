#ifndef WMMA_UTILS_H
#define WMMA_UTILS_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

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


template<typename Layout>
struct MatrixUtil;

template<>
struct MatrixUtil<row_major>
{
    template<typename DataT>
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

    template<typename DataT>
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
                mat[val] = val % 2 ? -val : val;
            }
        }
    }
};

template<>
struct MatrixUtil<col_major>
{
    template<typename DataT>
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

    template<typename DataT>
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
                mat[j * n + i] = val % 2 ? -val : val;
            }
        }
    }
};

template<int M, int N, int K>
void validateC(std::vector<float> const& a, std::vector<float> const& b, std::vector<float> const& c)
{
    for(int i=0; i < M; i++)
    {
        auto rowStartA = i*K;
        for(int j=0; j < N; j++)
        {
            auto colStartB = j;
            float result = 0.0f;
            for(int k = 0; k < K; k++)
            {
                result += (a[rowStartA + k] * b[colStartB + k*N]);
            } 

            if(c[i*M + j] != result)
            {
                std::cout << "No match: C( " << i << ", " << j << " )\n";
                std::cout << "(Expected, actual): ( " << result << ", " << c[i*M + j] << ")\n";
            }
        }
    }
}

#endif // WMMA_UTILS_H
