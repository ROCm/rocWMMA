/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#ifndef WMMA_TEST_COMMON_H
#define WMMA_TEST_COMMON_H

#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

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

template <uint32_t N>
using I = std::integral_constant<uint32_t, N>;

template <class... Ts, class F>
void for_each(std::tuple<Ts...>, F f)
{
    std::initializer_list<int> _ = {(f(Ts{}), 0)...}; // poor man's fold expression for C++11/14
    // (f(Ts{}), ...); // fold expression is for C++17 only
}

namespace quirks
{
    // rocBLAS does not yet support Ti/To/Tc = bf16/bf16/bf16
    template <typename InputT, typename OutputT, typename ComputeT>
    struct rocblas_supported : std::true_type
    {
    };

    template <>
    struct rocblas_supported<bfloat16_t, bfloat16_t, bfloat16_t> : std::false_type
    {
    };

    template <>
    struct rocblas_supported<int8_t, int8_t, int32_t> : std::false_type
    {
    };

} // namespace quirks

template <typename Layout>
struct MatrixUtil
{
    template <typename DataT>
    __host__ static inline void
        print(DataT const* mat, uint32_t m, uint32_t n, std::ostream& stream = std::cout)
    {
        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

        for(int i = 0; i < m; ++i) // row
        {
            stream << "[ ";
            for(int j = 0; j < n; ++j) // col
            {
                // (Row, col)
                stream << mat[index(i, j, ld)] << " ";
            }
            stream << "]\n";
        }
        stream << "\n";
    }

    template <typename DataT>
    __host__ static inline void print(std::vector<DataT> const& mat,
                                      uint32_t                  m,
                                      uint32_t                  n,
                                      std::ostream&             stream = std::cout)
    {
        assert(mat.size() == n * m);
        print(mat.data(), m, n, stream);
    }

    template <typename DataT>
    __host__ static inline void
        fill_with_padding(DataT* mat, uint32_t m, uint32_t n, DataT padValue)
    {
        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

#pragma omp parallel for
        for(int i = 0; i < m; ++i) // row
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j) // col
            {
                auto idx = index(i, j, ld);
                if(i == 0 || j == 0 || i == m - 1 || j == n - 1)
                    mat[idx] = padValue;
                else
                {
                    // Count up in integers, in ascending order for each row.
                    auto value = (i * n + j) % 13;
                    mat[idx] = (value % 2) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
                }
            }
        }
    }

    template <typename DataT>
    __host__ static inline void
        fill_with_padding(std::vector<DataT>& mat, uint32_t m, uint32_t n, DataT padValue)
    {
        assert(mat.size() == n * m);
        fill_with_padding(mat.data(), m, n, padValue);
    }

    template <typename DataT>
    __host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
    {
        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

#pragma omp parallel for
        for(int i = 0; i < m; ++i) // row
        {
#pragma omp parallel for
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
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
    {
        assert(mat.size() == n * m);
        fill(mat.data(), m, n);
    }

    template <typename DataT>
    __host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n, DataT value)
    {
#pragma omp parallel for
        for(int i = 0; i < m * n; ++i) // row
        {
            mat[i] = value;
        }
    }

    template <typename DataT>
    __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n, DataT value)
    {
        assert(mat.size() == n * m);
        fill(mat.data(), m, n, value);
    }

    template <typename DataT>
    __host__ static inline void GenerateLayoutIds(DataT* data, int m, int n)
    {
        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
        auto ld    = std::is_same<Layout, row_major>::value ? n : m;

#pragma omp parallel for
        for(int i = 0; i < m; ++i) // row
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j) // col
            {
                auto idx  = index(i, j, ld);
                data[idx] = static_cast<DataT>(idx);
            }
        }
    }
};

template <typename DataT, typename DataLayout>
std::pair<bool, double> compareEqualPadded(std::vector<DataT> const& a,
                                           std::vector<DataT> const& b,
                                           int                       M,
                                           int                       N,
                                           DataT                     padValue,
                                           double                    tolerance = 10.0)
{
    bool retval;

    assert(a.size() == M * N && "A and B do not match size M x N");
    assert(b.size() == (M + 2) * (N + 2) && "A and B do not match size (M+2) x (N+2)");

    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto index = std::is_same<DataLayout, row_major>::value ? rowMjr : colMjr;
    auto ldA   = std::is_same<DataLayout, row_major>::value ? N : M;
    auto ldB   = std::is_same<DataLayout, row_major>::value ? N + 2 : M + 2;

#pragma omp parallel for
    for(int i = 0; i < M + 2; ++i) // row
    {
#pragma omp parallel for
        for(int j = 0; j < N + 2; ++j) // col
        {
            auto numerator = 0.0;
            auto divisor   = 0.0;

            if(i == 0 || j == 0 || i == M + 1 || j == N + 1)
            {

                auto idx       = index(i, j, ldB);
                auto numerator = fabs(toDouble(b[idx]) - toDouble(padValue));
                auto divisor   = fabs(toDouble(b[idx])) + fabs(toDouble(padValue)) + 1.0;
            }
            else
            {
                auto idxB      = index(i, j, ldB);
                auto idxA      = index(i - 1, j - 1, ldA);
                auto numerator = fabs(toDouble(a[idxA]) - toDouble(b[idxB]));
                auto divisor   = fabs(toDouble(a[idxA])) + fabs(toDouble(b[idxB])) + 1.0;
            }

            auto relative_error = numerator / divisor;

            if(relative_error > max_relative_error)
            {
                max_relative_error = relative_error;
            }
            // NaN: propagate the error and break
            else if(relative_error != relative_error)
            {
                max_relative_error = relative_error;
                i                  = M;
                j                  = N;
            }
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        retval = false;
    }
    else
    {
        retval = true;
    }
    return std::make_pair(retval, max_relative_error);
}

template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
std::pair<bool, double>
    compareEqual(TypeA const* a, TypeB const* b, int M, int N, double tolerance = 10.0)
{
    bool retval;
    int  lda = std::is_same<LayoutA, row_major>::value ? N : M;
    int  ldb = std::is_same<LayoutB, row_major>::value ? N : M;

    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDoubleA = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };
    auto toDoubleB = [](TypeB const& val) { return static_cast<double>(static_cast<float>(val)); };

#pragma omp parallel for
    for(int i = 0; i < M; ++i) // Row
    {
#pragma omp parallel for
        for(int j = 0; j < N; ++j) // Col
        {
            auto indexA = std::is_same<LayoutA, row_major>::value ? (i * lda + j) : (i + j * lda);
            auto indexB = std::is_same<LayoutB, row_major>::value ? (i * ldb + j) : (i + j * ldb);

            auto relative_error = fabs(toDoubleA(a[indexA]) - toDoubleB(b[indexB]))
                                  / (fabs(toDoubleA(a[indexA])) + fabs(toDoubleB(b[indexB])) + 1.0);

            if(relative_error > max_relative_error)
            {
                max_relative_error = relative_error;
            }
            // NaN: propagate the error and break
            else if(relative_error != relative_error)
            {
                max_relative_error = relative_error;
                i                  = M;
                j                  = N;
            }
        }
    }

    auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        retval = false;
    }
    else
    {
        retval = true;
    }

    return std::make_pair(retval, max_relative_error);
}

template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
std::pair<bool, double> compareEqual(
    std::vector<TypeA> const& a, std::vector<TypeB> const& b, int M, int N, double tolerance = 10.0)
{
    assert(a.size() == b.size() && "A and B are not the same size");
    assert(a.size() == M * N && "A and B do not match size M x N");
    return compareEqual<TypeA, TypeB, LayoutA, LayoutB>(a.data(), b.data(), M, N, tolerance);
}

#endif // WMMA_TEST_COMMON_H
