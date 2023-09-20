/* ************************************************************************
 * Copyright (c) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocwmma_xfloat32.h provides struct for rocwmma_xfloat32 typedef
 */

#ifndef ROCWMMA_XFLOAT32_HPP
#define ROCWMMA_XFLOAT32_HPP

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this is a C compiler, C++ compiler below C++11, or a host-only compiler, we only
// include a minimal definition of rocwmma_xfloat32

#include <stdint.h>
typedef struct
{
    float data;
} rocwmma_xfloat32;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#if !defined(__HIPCC_RTC__)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <ostream>
#include <type_traits>

#else

namespace std
{
    using __hip_internal::is_standard_layout;
    using __hip_internal::is_trivial;
}

#endif // !defined(__HIPCC_RTC__)

#include "config.hpp"

struct rocwmma_xfloat32
{
    float data;

    enum round_t
    {
        round_up
    };

    ROCWMMA_HOST_DEVICE rocwmma_xfloat32() = default;

    // round upper 19 bits of IEEE float to convert to xfloat32
    explicit ROCWMMA_HOST_DEVICE rocwmma_xfloat32(float f, round_t)
        : data(float_to_xfloat32(f))
    {
    }

    explicit ROCWMMA_HOST_DEVICE rocwmma_xfloat32(float f)
        : data(truncate_float_to_xfloat32(f))
    {
    }

    // zero extend lower 13 bits of xfloat32 to convert to IEEE float
    ROCWMMA_HOST_DEVICE operator float() const
    {
        return data;
    }

    explicit ROCWMMA_HOST_DEVICE operator bool() const
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {data};
        return u.int32 & 0x7fffe000;
    }

    explicit ROCWMMA_HOST_DEVICE operator uint32_t() const
    {
        return uint32_t(float(*this));
    }

    explicit ROCWMMA_HOST_DEVICE operator long() const
    {
        return long(float(*this));
    }

    explicit ROCWMMA_HOST_DEVICE operator double() const
    {
        return double(float(*this));
    }

private:
    static ROCWMMA_HOST_DEVICE float float_to_xfloat32(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the xfloat32 mantissa up by adding 0xFFF, plus
            // 1 if the least significant bit of the xfloat32 mantissa is 1 (odd).
            // This causes the xfloat32's mantissa to be incremented by 1 if the 13
            // least significant bits of the float mantissa are greater than 0x1000,
            // or if they are equal to 0x1000 and the least significant bit of the
            // xfloat32 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 13 bits are exactly 0x1000. If the xfloat32 mantissa already
            // has the value 0x3ff, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded xfloat32 value. When the xfloat32 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x3FF, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the xfloat32 value has an exponent of 0xFE and a mantissa of 0x3FF,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.

            u.int32 += 0xfff + ((u.int32 >> 13) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0x1fff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 13 bits of the mantissa are 1, we set the least significant bit
            // of the xfloat32 mantissa, in order to preserve signaling NaN in case
            // the xfloat32's mantissa bits are all 0.
            u.int32 |= 0x2000; // Preserve signaling NaN
        }

        u.int32 &= 0xffffe000;
        return u.fp32;
    }

    // Truncate instead of rounding
    static ROCWMMA_HOST_DEVICE float truncate_float_to_xfloat32(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};

        u.int32 = u.int32 & 0xffffe000;
        return u.fp32;
    }
};

typedef struct
{
    float data;
} rocwmma_xfloat32_public;

static_assert(std::is_standard_layout<rocwmma_xfloat32>{},
              "rocwmma_xfloat32 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<rocwmma_xfloat32>{},
              "rocwmma_xfloat32 is not a trivial type, and thus is "
              "incompatible with C.");

#if !defined(__HIPCC_RTC__)
static_assert(sizeof(rocwmma_xfloat32) == sizeof(rocwmma_xfloat32_public)
                  && offsetof(rocwmma_xfloat32, data) == offsetof(rocwmma_xfloat32_public, data),
              "internal rocwmma_xfloat32 does not match public rocwmma_xfloat32");
#endif // !defined(__HIPCC_RTC__)

#if !defined(__HIPCC_RTC__)
inline std::ostream& operator<<(std::ostream& os, const rocwmma_xfloat32& xf32)
{
    return os << float(xf32);
}
#endif // !defined(__HIPCC_RTC__)
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator+(rocwmma_xfloat32 a)
{
    return a;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator-(rocwmma_xfloat32 a)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {a.data};
    u.int32 ^= 0x80000000;
    return rocwmma_xfloat32(u.fp32);
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator+(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return rocwmma_xfloat32(float(a) + float(b));
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator-(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return rocwmma_xfloat32(float(a) - float(b));
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator*(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return rocwmma_xfloat32(float(a) * float(b));
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator/(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return rocwmma_xfloat32(float(a) / float(b));
}
inline ROCWMMA_HOST_DEVICE bool operator<(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return float(a) < float(b);
}
inline ROCWMMA_HOST_DEVICE bool operator==(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return float(a) == float(b);
}
inline ROCWMMA_HOST_DEVICE bool operator>(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return b < a;
}
inline ROCWMMA_HOST_DEVICE bool operator<=(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return !(a > b);
}
inline ROCWMMA_HOST_DEVICE bool operator!=(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return !(a == b);
}
inline ROCWMMA_HOST_DEVICE bool operator>=(rocwmma_xfloat32 a, rocwmma_xfloat32 b)
{
    return !(a < b);
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator+=(rocwmma_xfloat32& a, rocwmma_xfloat32 b)
{
    return a = a + b;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator-=(rocwmma_xfloat32& a, rocwmma_xfloat32 b)
{
    return a = a - b;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator*=(rocwmma_xfloat32& a, rocwmma_xfloat32 b)
{
    return a = a * b;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator/=(rocwmma_xfloat32& a, rocwmma_xfloat32 b)
{
    return a = a / b;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator++(rocwmma_xfloat32& a)
{
    return a += rocwmma_xfloat32(1.0f);
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32& operator--(rocwmma_xfloat32& a)
{
    return a -= rocwmma_xfloat32(1.0f);
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator++(rocwmma_xfloat32& a, int)
{
    rocwmma_xfloat32 orig = a;
    ++a;
    return orig;
}
inline ROCWMMA_HOST_DEVICE rocwmma_xfloat32 operator--(rocwmma_xfloat32& a, int)
{
    rocwmma_xfloat32 orig = a;
    --a;
    return orig;
}

namespace std
{
    constexpr ROCWMMA_HOST_DEVICE bool isinf(rocwmma_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return !(~u.int32 & 0x7f800000) && !(u.int32 & 0x7fe000);
    }
    constexpr ROCWMMA_HOST_DEVICE bool isnan(rocwmma_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return !(~u.int32 & 0x7f800000) && +(u.int32 & 0x7fe000);
    }
    constexpr ROCWMMA_HOST_DEVICE bool iszero(rocwmma_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return (u.fp32 == 0.0f);
    }

    ROCWMMA_HOST_DEVICE inline rocwmma_xfloat32 sin(rocwmma_xfloat32 a)
    {
        return rocwmma_xfloat32(sinf(float(a)));
    }
    ROCWMMA_HOST_DEVICE inline rocwmma_xfloat32 cos(rocwmma_xfloat32 a)
    {
        return rocwmma_xfloat32(cosf(float(a)));
    }

    ROCWMMA_HOST_DEVICE constexpr rocwmma_xfloat32 real(const rocwmma_xfloat32& a)
    {
        return a;
    }
}

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // ROCWMMA_XFLOAT32_HPP
