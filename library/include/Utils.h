#ifndef WMMA_UTILS_H
#define WMMA_UTILS_H

#include <array>
#include <assert.h>
#include <iostream>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Constants.h"
#include "Types.h"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

struct Fp16Bits
{
    union
    {
        uint16_t      i16;
        float16_t     f16;
        hfloat16_t    h16;
        bfloat16_t    b16;
        unsigned char c16[16];
    };
    constexpr Fp16Bits(uint16_t initVal)
        : i16(initVal)
    {
    }
    constexpr Fp16Bits(float16_t initVal)
        : f16(initVal)
    {
    }
    constexpr Fp16Bits(hfloat16_t initVal)
        : h16(initVal)
    {
    }
    constexpr Fp16Bits(bfloat16_t initVal)
        : b16(initVal)
    {
    }
};

// Define std::numeric_limits<float16_t/hfloat16_t> functions that we need for validation
namespace std
{
    template <>
    __host__ __device__ constexpr float16_t numeric_limits<float16_t>::epsilon() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr float16_t numeric_limits<float16_t>::min() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr hfloat16_t numeric_limits<hfloat16_t>::epsilon() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr hfloat16_t numeric_limits<hfloat16_t>::min() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr bfloat16_t numeric_limits<bfloat16_t>::epsilon() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x3C00));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr bfloat16_t numeric_limits<bfloat16_t>::min() noexcept
    {
        ::Fp16Bits eps(static_cast<uint16_t>(0x007F));
        return eps.b16;
    }
}

// Define host side hfloat16_t operators that we need for validation

// Needed for compareEqual
__host__ inline bool operator==(const hfloat16_t& x, const hfloat16_t& y)
{
    auto absDiff = std::fabs(__half2float(x) - __half2float(y));
    auto absAdd  = std::fabs(__half2float(x) + __half2float(y));
    return absDiff <= __half2float(std::numeric_limits<hfloat16_t>::epsilon()) * absAdd * 2.0f
           || absDiff < __half2float(std::numeric_limits<hfloat16_t>::min());
}

__host__ inline bool operator!=(const hfloat16_t& x, const hfloat16_t& y)
{
    return !(x == y);
}

// Needed for MatrixUtil::fill
__host__ inline hfloat16_t operator-(const hfloat16_t& x)
{
    Fp16Bits fp16(x);
    fp16.i16 ^= 0x8000; // Flip sign
    return fp16.h16;
}

__host__ inline hfloat16_t operator*(const hfloat16_t& x, const hfloat16_t& y)
{
    return static_cast<hfloat16_t>(static_cast<float16_t>(x) * static_cast<float16_t>(y));
}

__host__ inline hfloat16_t operator+(const hfloat16_t& x, const hfloat16_t& y)
{
    return static_cast<hfloat16_t>(static_cast<float16_t>(x) + static_cast<float16_t>(y));
}

__host__ inline hfloat16_t& operator+=(hfloat16_t& x, const hfloat16_t& y)
{
    return x = static_cast<hfloat16_t>(static_cast<float16_t>(x) + static_cast<float16_t>(y));
}

template <typename Layout>
struct MatrixUtil
{
    template <typename DataT>
    __host__ static inline void print(std::vector<DataT> const& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);

        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

        for(int i = 0; i < m; ++i) // row
        {
            std::cout << "[ ";
            for(int j = 0; j < n; ++j) // col
            {
                // (Row, col)
                std::cout << mat[index(i, j, ld)] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);

        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

        for(int i = 0; i < m; ++i) // row
        {
            for(int j = 0; j < n; ++j) // col
            {
                // Count up in integers, in ascending order for each row.
                auto value = (i * n + j) % 13;
                auto idx   = index(i, j, ld);
                mat[idx]   = (value % 2) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
            }
        }
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n, DataT value)
    {
        assert(mat.size() == n * m);
        for(int i = 0; i < m * n; ++i) // row
        {
            mat[i] = value;
        }
    }
};

template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
void compareEqual(
    std::vector<TypeA> const& a, std::vector<TypeB> const& b, int M, int N, double tolerance = 10.0)
{
    assert(a.size() == b.size() && "A and B are not the same size");
    assert(a.size() == M * N && "A and B do not match size M x N");
    int lda = std::is_same<LayoutA, row_major>::value ? N : M;
    int ldb = std::is_same<LayoutB, row_major>::value ? N : M;

    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDoubleA = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };
    auto toDoubleB = [](TypeB const& val) { return static_cast<double>(static_cast<float>(val)); };

#pragma omp parallel for
    for(int i = 0; i < M; ++i) // Row
    {
        for(int j = 0; j < N; ++j) // Col
        {
            auto indexA = std::is_same<LayoutA, row_major>::value ? (i * lda + j) : (i + j * lda);
            auto indexB = std::is_same<LayoutB, row_major>::value ? (i * ldb + j) : (i + j * ldb);

            auto relative_error
                = (a[indexA] != static_cast<TypeA>(0))
                      ? fabs((toDoubleA(a[indexA]) - toDoubleB(b[indexB])) / toDoubleA(a[indexA]))
                      : 0.0;
            if(relative_error > max_relative_error)
            {
                max_relative_error = relative_error;
            }
        }
    }

    auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL: ";
    }
    else
    {
        std::cout << "PASS: ";
    }
    std::cout << "max_relative_error = " << max_relative_error << std::endl;
}

template <typename DataT>
void compareEqual(
    std::vector<DataT> const& a, std::vector<DataT> const& b, int M, int N, double tolerance = 10.0)
{
    assert(a.size() == b.size() && "A and B are not the same size");
    assert(a.size() == M * N && "A and B do not match size M x N");

    double   max_relative_error = 0.0;
    uint32_t numElements        = M * N;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

#pragma omp parallel for
    for(int i = 0; i < numElements; ++i)
    {
        auto relative_error = a[i] != static_cast<DataT>(0)
                                  ? fabs((toDouble(a[i]) - toDouble(b[i])) / toDouble(a[i]))
                                  : 0.0;
        if(relative_error > max_relative_error)
        {
            max_relative_error = relative_error;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL: ";
    }
    else
    {
        std::cout << "PASS: ";
    }
    std::cout << "max_relative_error = " << max_relative_error << std::endl;
}

template <typename DataT>
constexpr const char* dataTypeToString()
{
    if(std::is_same<DataT, float16_t>::value)
    {
        return "f16";
    }
    else if(std::is_same<DataT, hfloat16_t>::value)
    {
        return "h16";
    }
    else if(std::is_same<DataT, bfloat16_t>::value)
    {
        return "bf16";
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
