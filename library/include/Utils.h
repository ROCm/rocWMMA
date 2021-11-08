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
#ifndef WMMA_UTILS_H
#define WMMA_UTILS_H

#include <array>
#include <assert.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"

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
#if !(__cplusplus >= 201703L)
    template <typename F, typename Tuple, size_t... I>
    auto apply_impl(F fn, Tuple t, std::index_sequence<I...>)
    {
        return fn(std::get<I>(t)...);
    }
    template <typename F, typename Tuple>
    auto apply(F fn, Tuple t)
    {
        const std::size_t size = std::tuple_size<Tuple>::value;
        return apply_impl(fn, t, std::make_index_sequence<size>());
    }
#endif
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

template <typename DataT>
constexpr const char* dataTypeToString()
{
    return "invalid";
}

template <>
constexpr const char* dataTypeToString<float16_t>()
{
    return "f16";
}

template <>
constexpr const char* dataTypeToString<hfloat16_t>()
{
    return "h16";
}

template <>
constexpr const char* dataTypeToString<bfloat16_t>()
{
    return "bf16";
}

template <>
constexpr const char* dataTypeToString<float32_t>()
{
    return "f32";
}

template <>
constexpr const char* dataTypeToString<float64_t>()
{
    return "f64";
}

template <>
constexpr const char* dataTypeToString<int8_t>()
{
    return "i8";
}

template <>
constexpr const char* dataTypeToString<uint8_t>()
{
    return "u8";
}

template <>
constexpr const char* dataTypeToString<int32_t>()
{
    return "i32";
}

template <>
constexpr const char* dataTypeToString<uint32_t>()
{
    return "u32";
}

template <>
constexpr const char* dataTypeToString<row_major>()
{
    return "T";
}

template <>
constexpr const char* dataTypeToString<col_major>()
{
    return "N";
}

namespace std
{
    template <typename T>
    __device__ inline pair<T, T> reverse(pair<T, T> const& p)
    {
        return make_pair(p.second, p.first);
    }

    inline pair<uint32_t, uint32_t> operator+(pair<uint32_t, uint32_t> const& lhs,
                                              pair<uint32_t, uint32_t> const& rhs)
    {
        return make_pair(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }

    inline pair<uint32_t, uint32_t>& operator+=(pair<uint32_t, uint32_t>&       lhs,
                                                pair<uint32_t, uint32_t> const& rhs)
    {
        get<0>(lhs) += get<0>(rhs);
        get<1>(lhs) += get<1>(rhs);
        return lhs;
    }

    inline pair<uint32_t, uint32_t> operator-(pair<uint32_t, uint32_t> const& lhs,
                                              pair<uint32_t, uint32_t> const& rhs)
    {
        return make_pair(get<0>(lhs) - get<0>(rhs), get<1>(lhs) - get<1>(rhs));
    }

    inline pair<uint32_t, uint32_t>& operator-=(pair<uint32_t, uint32_t>&       lhs,
                                                pair<uint32_t, uint32_t> const& rhs)
    {
        get<0>(lhs) -= get<0>(rhs);
        get<1>(lhs) -= get<1>(rhs);
        return lhs;
    }
} // namespace std

#endif // WMMA_UTILS_H
