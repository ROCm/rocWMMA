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
                mat[val]    = (val % 2 ? -val : val) % 13;
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
                mat[i + j * m] = (val % 2 ? -val : val) % 13;
            }
        }
    }
};

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
            ComputeT accum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                auto indexA
                    = std::is_same<LayoutA, row_major>::value ? (i * lda + k) : (i + lda * k);
                auto indexB
                    = std::is_same<LayoutB, row_major>::value ? (k * ldb + j) : (k + j * ldb);
                accum += static_cast<ComputeT>(a[indexA]) * static_cast<ComputeT>(b[indexB]);
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

    double max_relative_error = 0;

#pragma omp parallel for
    for(int i = 0; i < M; ++i) // Row
    {
        for(int j = 0; j < N; ++j) // Col
        {
            auto indexA = std::is_same<LayoutA, row_major>::value ? (i * lda + j) : (i + j * lda);
            auto indexB = std::is_same<LayoutB, row_major>::value ? (i * ldb + j) : (i + j * ldb);

            auto relative_error = fabs(double(a[indexA] - b[indexB]) / a[indexA]);
            if(relative_error > max_relative_error)
            {
                max_relative_error = relative_error;
            }
        }
    }

    auto eps       = std::numeric_limits<TypeA>::epsilon();
    auto tolerance = 10.0;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
        std::cout << "FAIL: ";
    else
        std::cout << "PASS: ";
    std::cout << "max_relative_error = " << max_relative_error << std::endl;
}

template <typename DataT>
struct MfmaPerfTraits;

template <>
struct MfmaPerfTraits<float16_t>
{
    enum : uint32_t
    {
        Multiplier = 16
    };
};

template <>
struct MfmaPerfTraits<float32_t>
{
    enum : uint32_t
    {
        Multiplier = 4
    };
};

template <typename DataT>
struct PerfTraits;

template <>
struct PerfTraits<float16_t>
{
    enum : uint32_t
    {
        Multiplier = 4
    };
};

template <>
struct PerfTraits<float32_t>
{
    enum : uint32_t
    {
        Multiplier = 2
    };
};

class Mi100;

template <typename GfxArch>
struct HardwareTraits;

template <>
struct HardwareTraits<Mi100>
{
    enum : uint32_t
    {
        CuCount = 120,
    };
};

template <typename DataT,
          typename GfxArch,
          template <typename> class PerformanceTraits = MfmaPerfTraits>
inline double calculatePeakGFlops(uint32_t freqMHz)
{
    auto basePeakGFlops
        = static_cast<double>(64.0 * HardwareTraits<GfxArch>::CuCount * freqMHz) / 1000.0;
    auto multiplier = (double)(PerformanceTraits<DataT>::Multiplier);
    return multiplier * basePeakGFlops;
}

inline double calculateGFlops(uint32_t M, uint32_t N, uint32_t K, double elapsedTimeMs)
{
    constexpr double flopsPerMac = 2.0;
    return static_cast<double>(flopsPerMac * M * N * K) / 1000000.0 / elapsedTimeMs;
}

template <typename DataT>
constexpr const char* dataTypeToString()
{
    if(std::is_same<DataT, float16_t>::value)
    {
        return "f16";
    }
    else if(std::is_same<DataT, float32_t>::value)
    {
        return "f32";
    }
    else if(std::is_same<DataT, int32_t>::value)
    {
        return "i32";
    }
    else
    {
        return "invalid";
    }
}

#endif // WMMA_UTILS_H
