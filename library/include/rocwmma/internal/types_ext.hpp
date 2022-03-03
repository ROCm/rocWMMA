/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_TYPES_EXT_HPP
#define ROCWMMA_TYPES_EXT_HPP

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <limits>
#include <ostream>
#include <type_traits>

#include "types.hpp"

/**
 * \ingroup rocwmma
 * \defgroup DataTypes
 *
 * @brief Definition and metadata on supported data types of matrices.
 *
 * Native Data Types:
 * float64_t = f64 = double
 * float = f32
 * _Float16 = f16
 * int8
 * uint8
 * int16
 * int32
 * uint32
 *
 *
 * Non-Native Data Types:
 * h16 = __half
 * bf16 = bfloat16
 *
 */

namespace rocwmma
{
    namespace detail
    {
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

    } // namespace detail

} // namespace rocwmma

namespace std
{
    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(float16_t)  //////////
    ///////////////////////////////////////////////////////////

    inline ostream& operator<<(ostream& stream, rocwmma::float16_t const& val)
    {
        return stream << static_cast<float>(val);
    }

    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(hfloat16_t)  /////////
    ///////////////////////////////////////////////////////////

    inline ostream& operator<<(ostream& stream, rocwmma::hfloat16_t const& val)
    {
        return stream << __half2float(val);
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<float16_t>  //////////////
    ///////////////////////////////////////////////////////////

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.f16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::float16_t
             numeric_limits<rocwmma::float16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.f16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<hfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.h16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::hfloat16_t
             numeric_limits<rocwmma::hfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.h16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<bfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x3C00));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F80));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFF7F));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F7F));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x007F));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

    template <>
    __host__ __device__ constexpr rocwmma::bfloat16_t
             numeric_limits<rocwmma::bfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

} // namespace std

namespace rocwmma
{
    ///////////////////////////////////////////////////////////
    ///////////  rocwmma::hfloat16_t host operators  //////////
    ///////////////////////////////////////////////////////////

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

    __host__ inline hfloat16_t operator-(const hfloat16_t& x)
    {
        detail::Fp16Bits fp16(x);
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

    ///////////////////////////////////////////////////////////
    ///////////  rocwmma::dataTypeToString overloads  /////////
    ///////////////////////////////////////////////////////////

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

    template<typename T,
    typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
    constexpr auto maxExactInteger() -> decltype(std::numeric_limits<T>::max()) 
    {
        return std::numeric_limits<T>::max();
    }

    template<typename T,
    typename std::enable_if_t<std::is_floating_point<T>::value && std::numeric_limits<T>::digits, int> = 0>
    constexpr auto maxExactInteger() -> typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>
    {
        using RetT = typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>;
        return ((RetT)1 << std::numeric_limits<T>::digits);
    }

    template<typename T,
    typename std::enable_if_t<std::is_same<T, hfloat16_t>::value || std::is_same<T, float16_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f16 mantissa is 10 bits
        return ((int32_t)1 << 11);
    }

    template<typename T,
    typename std::enable_if_t<std::is_same<T, bfloat16_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // b16 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }

} // namespace rocwmma

#endif // ROCWMMA_TYPES_EXT_HPP
