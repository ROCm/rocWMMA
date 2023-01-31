/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#if !defined(__HIPCC_RTC__)
#include <cmath>
#include <hip/hip_bfloat16.h>
#include <limits>
#include <ostream>
#else
#include "utils.hpp"
#endif // !defined(__HIPCC_RTC__)

// #include <hip/hip_bfloat16.h>
// #include <hip/hip_fp16.h>

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

#if !defined(__HIPCC_RTC__)
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

    __host__ inline hfloat16_t operator+(const hfloat16_t& x, const hfloat16_t& y)
    {
        return static_cast<hfloat16_t>(static_cast<float16_t>(x) + static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t operator-(const hfloat16_t& x, const hfloat16_t& y)
    {
        return static_cast<hfloat16_t>(static_cast<float16_t>(x) - static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t operator*(const hfloat16_t& x, const hfloat16_t& y)
    {
        return static_cast<hfloat16_t>(static_cast<float16_t>(x) * static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t operator/(const hfloat16_t& x, const hfloat16_t& y)
    {
        return static_cast<hfloat16_t>(static_cast<float16_t>(x) / static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t& operator+=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = static_cast<hfloat16_t>(static_cast<float16_t>(x) + static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t& operator-=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = static_cast<hfloat16_t>(static_cast<float16_t>(x) - static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t& operator*=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = static_cast<hfloat16_t>(static_cast<float16_t>(x) * static_cast<float16_t>(y));
    }

    __host__ inline hfloat16_t& operator/=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = static_cast<hfloat16_t>(static_cast<float16_t>(x) / static_cast<float16_t>(y));
    }
#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

namespace std
{
#if !defined(__HIPCC_RTC__)
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
#endif // !defined(__HIPCC_RTC__)

} // namespace std

#endif // ROCWMMA_TYPES_EXT_HPP
