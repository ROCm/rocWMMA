/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_FLOAT8_H
#define ROCWMMA_FLOAT8_H

#include "config.hpp"

#if defined(__HIPCC_RTC__)

using uint8_t = __hip_internal::uint8_t;
using uint16_t = __hip_internal::uint16_t;

namespace std
{
    template <bool B, class T, class F>
    struct conditional;
}

#endif

// We are clipping in down conversion by default
#define rocwmma_F8_downcast_clipping 1

namespace rocwmma_hip_f8_impl
{
    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    ROCWMMA_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

    template <int wm, int we, typename T, bool negative_zero_nan>
    ROCWMMA_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace rocwmma_hip_f8_impl

#include "rocwmma_hip_f8_impl.h"

static ROCWMMA_DEVICE bool rocwmma_hip_f8_bias_mode_bit_device = true;
static bool                rocwmma_hip_f8_bias_mode_bit_host   = true;

struct rocwmma_f8
{
    uint8_t data;
    enum class rocwmma_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    ROCWMMA_HOST_DEVICE rocwmma_f8() = default;

#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    // device specific optimized F8 down-conversion code
    template <bool stochastic_rounding = false>
    static ROCWMMA_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef rocwmma_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
        {
            val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
        }
#endif
        if(stochastic_rounding)
        {
            ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
            val.i32val = ival;
            i8data     = val.i8val[0]; // little endian
        }
        else // RNE CVT
        {
            ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                val.fval, val.fval, ival, false); // false -> WORD0
            val.i32val = ival;
            i8data     = val.i8val[0];
        }
        return i8data;
    }

#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

    // constructor from float
#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

    // NOTE: ON-DEVICE... always optimal bias
    explicit ROCWMMA_DEVICE rocwmma_f8(float                        v,
                                       rocwmma_hip_f8_rounding_mode rm
                                       = rocwmma_hip_f8_rounding_mode::standard,
                                       uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == rocwmma_hip_f8_rounding_mode::stochastic)
        {
            data = cast_to_f8_from_f32<true>(v, rng);
        }
        else
        {
            data = cast_to_f8_from_f32<false>(v);
        }
    }

    // Host only implementation using s/w simulation
    explicit ROCWMMA_HOST
#else
    // both Host and DEVICE for non-gfx940 using s/w simulation
    explicit ROCWMMA_HOST_DEVICE
#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
        rocwmma_f8(float                        v,
                   rocwmma_hip_f8_rounding_mode rm  = rocwmma_hip_f8_rounding_mode::standard,
                   uint32_t                     rng = 0)
    {
#ifdef rocwmma_F8_downcast_clipping
        data = rocwmma_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == rocwmma_hip_f8_rounding_mode::stochastic), rng);
#else // rocwmma_F8_downcast_clipping
        data = rocwmma_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == rocwmma_hip_f8_rounding_mode::stochastic), rng);
#endif // rocwmma_F8_downcast_clipping
    }

    // Constructor from half
    explicit ROCWMMA_HOST_DEVICE rocwmma_f8(_Float16                     v,
                                            rocwmma_hip_f8_rounding_mode rm
                                            = rocwmma_hip_f8_rounding_mode::standard,
                                            uint32_t rng = 0)
        : rocwmma_f8((float)v, rm, rng)

    {
    }

    // constructor from int
    explicit ROCWMMA_HOST_DEVICE rocwmma_f8(int                          v,
                                            rocwmma_hip_f8_rounding_mode rm
                                            = rocwmma_hip_f8_rounding_mode::standard,
                                            uint32_t rng = 0)
        : rocwmma_f8((float)v, rm, rng)
    {
    }

    // constructor from unsigned int
    explicit ROCWMMA_HOST_DEVICE rocwmma_f8(unsigned int                 v,
                                            rocwmma_hip_f8_rounding_mode rm
                                            = rocwmma_hip_f8_rounding_mode::standard,
                                            uint32_t rng = 0)
        : rocwmma_f8((float)v, rm, rng)
    {
    }

    // constructor from double
    explicit ROCWMMA_HOST_DEVICE rocwmma_f8(double                       v,
                                            rocwmma_hip_f8_rounding_mode rm
                                            = rocwmma_hip_f8_rounding_mode::standard,
                                            uint32_t rng = 0)
        : rocwmma_f8((float)v, rm, rng)
    {
    }

    // convert to float
#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    // upcast using device specific intrinsic
    explicit inline ROCWMMA_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline ROCWMMA_HOST operator float() const
#else // non gfx940
    explicit inline ROCWMMA_HOST_DEVICE operator float() const
#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    {
        return rocwmma_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline ROCWMMA_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to unsigned int
    explicit inline ROCWMMA_HOST_DEVICE operator uint32_t() const
    {
        return uint32_t(float(*this)); // convert to float, then convert to u32
    }

    // convert to long
    explicit inline ROCWMMA_HOST_DEVICE operator long() const
    {
        return long(float(*this)); // convert to float, then convert to long
    }

    // convert to double
    explicit inline ROCWMMA_HOST_DEVICE operator double() const
    {
        return double(float(*this)); // convert to float, then convert to double
    }

    inline ROCWMMA_HOST_DEVICE rocwmma_f8 operator-()
    {
        this->data ^= 0x80;
        return *this;
    }

    // check for zero
    inline ROCWMMA_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline ROCWMMA_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline ROCWMMA_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }
};

struct rocwmma_bf8
{
    uint8_t data;
    enum class rocwmma_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    ROCWMMA_HOST_DEVICE rocwmma_bf8() = default;

#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static ROCWMMA_DEVICE uint8_t cast_to_bf8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef rocwmma_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
        {
            val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
        }
#endif
        if(stochastic_rounding)
        {
            ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
            val.i32val = ival;
            i8data     = val.i8val[0]; // little endian
        }
        else // RNE CVT
        {
            ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                val.fval, val.fval, ival, false); // false -> WORD0
            val.i32val = ival;
            i8data     = val.i8val[0];
        }
        return i8data;
    }

#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

    // constructor from float
#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

    // NOTE: ON-DEVICE... always optimal bias
    explicit ROCWMMA_DEVICE rocwmma_bf8(float                        v,
                                        rocwmma_hip_f8_rounding_mode rm
                                        = rocwmma_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == rocwmma_hip_f8_rounding_mode::stochastic)
        {
            data = cast_to_bf8_from_f32<true>(v, rng);
        }
        else
        {
            data = cast_to_bf8_from_f32<false>(v);
        }
    }

    // Host only implementation using s/w simulation
    explicit ROCWMMA_HOST
#else
    // both Host and DEVICE for non-gfx940 using s/w simulation
    explicit ROCWMMA_HOST_DEVICE
#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
        rocwmma_bf8(float                        v,
                    rocwmma_hip_f8_rounding_mode rm  = rocwmma_hip_f8_rounding_mode::standard,
                    uint32_t                     rng = 0)
    {
#ifdef rocwmma_F8_downcast_clipping
        data = rocwmma_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == rocwmma_hip_f8_rounding_mode::stochastic), rng);
#else
        data = rocwmma_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == rocwmma_hip_f8_rounding_mode::stochastic), rng);
#endif // rocwmma_F8_downcast_clipping
    }

    // Constructor from half
    explicit ROCWMMA_HOST_DEVICE rocwmma_bf8(_Float16                     v,
                                             rocwmma_hip_f8_rounding_mode rm
                                             = rocwmma_hip_f8_rounding_mode::standard,
                                             uint32_t rng = 0)
        : rocwmma_bf8((float)v, rm, rng)
    {
    }

    // constructor from int
    explicit ROCWMMA_HOST_DEVICE rocwmma_bf8(int                          v,
                                             rocwmma_hip_f8_rounding_mode rm
                                             = rocwmma_hip_f8_rounding_mode::standard,
                                             uint32_t rng = 0)
        : rocwmma_bf8((float)v, rm, rng)
    {
    }

    // constructor from unsigned int
    explicit ROCWMMA_HOST_DEVICE rocwmma_bf8(unsigned int                 v,
                                             rocwmma_hip_f8_rounding_mode rm
                                             = rocwmma_hip_f8_rounding_mode::standard,
                                             uint32_t rng = 0)
        : rocwmma_bf8((float)v, rm, rng)
    {
    }

    // constructor from double
    explicit ROCWMMA_HOST_DEVICE rocwmma_bf8(double                       v,
                                             rocwmma_hip_f8_rounding_mode rm
                                             = rocwmma_hip_f8_rounding_mode::standard,
                                             uint32_t rng = 0)
        : rocwmma_bf8((float)v, rm, rng)
    {
    }

    // convert to float
#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    // upcast using device specific intrinsic
    explicit inline ROCWMMA_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }
    explicit inline ROCWMMA_HOST operator float() const
#else // non gfx940
    explicit inline ROCWMMA_HOST_DEVICE operator float() const
#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    {
        return rocwmma_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline ROCWMMA_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to unsigned int
    explicit inline ROCWMMA_HOST_DEVICE operator uint32_t() const
    {
        return uint32_t(float(*this)); // convert to float, then convert to u32
    }

    // convert to long
    explicit inline ROCWMMA_HOST_DEVICE operator long() const
    {
        return long(float(*this)); // convert to float, then convert to long
    }

    // convert to double
    explicit inline ROCWMMA_HOST_DEVICE operator double() const
    {
        return double(float(*this)); // convert to float, then convert to double
    }

    inline ROCWMMA_HOST_DEVICE rocwmma_bf8 operator-()
    {
        this->data ^= 0x80;
        return *this;
    }

    // check for zero
    inline ROCWMMA_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline ROCWMMA_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline ROCWMMA_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }
};

namespace std
{
    ROCWMMA_HOST_DEVICE inline rocwmma_f8 sin(rocwmma_f8 a)
    {
        return rocwmma_f8(sinf(float(a)));
    }
    ROCWMMA_HOST_DEVICE inline rocwmma_f8 cos(rocwmma_f8 a)
    {
        return rocwmma_f8(cosf(float(a)));
    }
    ROCWMMA_HOST_DEVICE inline rocwmma_bf8 sin(rocwmma_bf8 a)
    {
        return rocwmma_bf8(sinf(float(a)));
    }
    ROCWMMA_HOST_DEVICE inline rocwmma_bf8 cos(rocwmma_bf8 a)
    {
        return rocwmma_bf8(cosf(float(a)));
    }
    ROCWMMA_HOST_DEVICE constexpr rocwmma_f8 real(const rocwmma_f8& a)
    {
        return a;
    }
    ROCWMMA_HOST_DEVICE constexpr rocwmma_bf8 real(const rocwmma_bf8& a)
    {
        return a;
    }
}

#if !defined(__HIPCC_RTC__)

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const rocwmma_f8& f8)
{
    return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const rocwmma_bf8& bf8)
{
    return os << float(bf8);
}

#endif // !defined(__HIPCC_RTC__)

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline ROCWMMA_HOST_DEVICE float operator+(const float fa, rocwmma_f8 b)
{
    return (fa + float(b));
}

inline ROCWMMA_HOST_DEVICE float operator+(const float fa, rocwmma_bf8 b)
{
    return (fa + float(b));
}

inline ROCWMMA_HOST_DEVICE float operator+(rocwmma_f8 a, const float fb)
{
    return (float(a) + fb);
}

inline ROCWMMA_HOST_DEVICE float operator+(rocwmma_bf8 a, const float fb)
{
    return (float(a) + fb);
}

inline ROCWMMA_HOST_DEVICE float operator+(rocwmma_f8 a, rocwmma_bf8 b)
{
    return (float(a) + float(b));
}

inline ROCWMMA_HOST_DEVICE float operator+(rocwmma_bf8 a, rocwmma_f8 b)
{
    return (float(a) + float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_f8 operator+(rocwmma_f8 a, rocwmma_f8 b)
{
    return rocwmma_f8(float(a) + float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_bf8 operator+(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return rocwmma_bf8(float(a) + float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_f8& operator+=(rocwmma_f8& a, rocwmma_f8 b)
{
    return a = rocwmma_f8(float(a) + float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_bf8& operator+=(rocwmma_bf8& a, rocwmma_bf8 b)
{
    return a = rocwmma_bf8(float(a) + float(b));
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline ROCWMMA_HOST_DEVICE float operator-(const float fa, rocwmma_f8 b)
{
    return (fa - float(b));
}

inline ROCWMMA_HOST_DEVICE float operator-(const float fa, rocwmma_bf8 b)
{
    return (fa - float(b));
}

inline ROCWMMA_HOST_DEVICE float operator-(rocwmma_f8 a, const float fb)
{
    return (float(a) - fb);
}

inline ROCWMMA_HOST_DEVICE float operator-(rocwmma_bf8 a, const float fb)
{
    return (float(a) - fb);
}

inline ROCWMMA_HOST_DEVICE float operator-(rocwmma_f8 a, rocwmma_bf8 b)
{
    return (float(a) - float(b));
}

inline ROCWMMA_HOST_DEVICE float operator-(rocwmma_bf8 a, rocwmma_f8 b)
{
    return (float(a) - float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_f8 operator-(rocwmma_f8 a, rocwmma_f8 b)
{
    return rocwmma_f8(float(a) - float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_bf8 operator-(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return rocwmma_bf8(float(a) - float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_f8& operator-=(rocwmma_f8& a, rocwmma_f8 b)
{
    return a = rocwmma_f8(float(a) - float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_bf8& operator-=(rocwmma_bf8& a, rocwmma_bf8 b)
{
    return a = rocwmma_bf8(float(a) - float(b));
}

// overloading multiplication, always returns float,
inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_f8 a, rocwmma_f8 b)
{
    return float(a) * float(b);
}

inline ROCWMMA_HOST_DEVICE float operator*(float a, rocwmma_f8 b)
{
    return (a * float(b));
}

inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_f8 a, float b)
{
    return (float(a) * b);
}

inline ROCWMMA_HOST_DEVICE float operator*(int32_t a, rocwmma_f8 b)
{
    return ((float)a * float(b));
}

inline ROCWMMA_HOST_DEVICE float operator*(double a, rocwmma_f8 b)
{
    return ((float)a * float(b));
}

inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return float(a) * float(b);
}

inline ROCWMMA_HOST_DEVICE float operator*(float a, rocwmma_bf8 b)
{
    return (a * float(b));
}

inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_bf8 a, float b)
{
    return (float(a) * b);
}

inline ROCWMMA_HOST_DEVICE float operator*(int32_t a, rocwmma_bf8 b)
{
    return ((float)a * float(b));
}

inline ROCWMMA_HOST_DEVICE float operator*(double a, rocwmma_bf8 b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_f8 a, rocwmma_bf8 b)
{
    return float(a) * float(b);
}

inline ROCWMMA_HOST_DEVICE float operator*(rocwmma_bf8 a, rocwmma_f8 b)
{
    return float(a) * float(b);
}

// overloading division, always returns float,
inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_f8 a, rocwmma_f8 b)
{
    return float(a) / float(b);
}

inline ROCWMMA_HOST_DEVICE float operator/(float a, rocwmma_f8 b)
{
    return (a / float(b));
}

inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_f8 a, float b)
{
    return (float(a) / b);
}

inline ROCWMMA_HOST_DEVICE float operator/(int32_t a, rocwmma_f8 b)
{
    return ((float)a / float(b));
}

inline ROCWMMA_HOST_DEVICE float operator/(double a, rocwmma_f8 b)
{
    return ((float)a / float(b));
}

inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return float(a) / float(b);
}

inline ROCWMMA_HOST_DEVICE float operator/(float a, rocwmma_bf8 b)
{
    return (a / float(b));
}

inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_bf8 a, float b)
{
    return (float(a) / b);
}

inline ROCWMMA_HOST_DEVICE float operator/(int32_t a, rocwmma_bf8 b)
{
    return ((float)a / float(b));
}

inline ROCWMMA_HOST_DEVICE float operator/(double a, rocwmma_bf8 b)
{
    return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_f8 a, rocwmma_bf8 b)
{
    return float(a) / float(b);
}

inline ROCWMMA_HOST_DEVICE float operator/(rocwmma_bf8 a, rocwmma_f8 b)
{
    return float(a) / float(b);
}

inline ROCWMMA_HOST_DEVICE rocwmma_f8& operator/=(rocwmma_f8& a, rocwmma_f8 b)
{
    return a = rocwmma_f8(float(a) / float(b));
}

inline ROCWMMA_HOST_DEVICE rocwmma_bf8& operator/=(rocwmma_bf8& a, rocwmma_bf8 b)
{
    return a = rocwmma_bf8(float(a) / float(b));
}

// overloading for compare
inline ROCWMMA_HOST_DEVICE bool operator==(rocwmma_f8 a, rocwmma_f8 b)
{
    return (a.data == b.data);
}

inline ROCWMMA_HOST_DEVICE bool operator==(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return (a.data == b.data);
}

inline ROCWMMA_HOST_DEVICE bool operator!=(rocwmma_f8 a, rocwmma_f8 b)
{
    return (a.data != b.data);
}

inline ROCWMMA_HOST_DEVICE bool operator!=(rocwmma_bf8 a, rocwmma_bf8 b)
{
    return (a.data != b.data);
}

// ================ Explicit downcasting to support different rounding (RNE, SR) ===============
// NOTE: we going to remove all assignment operator overloading from other types and enforce
// this explicit_downcast function to make any roudning behavior default
// We have to explicitly call this function with SR flag

template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<std::is_same<T, Ta>{}, int>::type = 0>
inline ROCWMMA_HOST_DEVICE T explicit_downcast(Ta a)
{
    // same type, no conversion
    return a;
}

// Use h/w intrinsic and optimized version when __gfx940__
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && (std::is_same<T, rocwmma_f8>{} || std::is_same<T, rocwmma_bf8>{})),
                            int>::type
    = 0>
inline ROCWMMA_HOST_DEVICE T explicit_downcast(Ta a, uint32_t rng)
{
#if ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
    // NOTE: we are directly calling cast_to_f8_from_f32 instead of constructor to optimize away one runtime branch
    T val;
    if(std::is_same<T, rocwmma_f8>::value)
    {
        val.data = rocwmma_f8::cast_to_f8_from_f32<stochastic_rounding>(float(a), rng);
    }
    else
    {
        val.data = rocwmma_bf8::cast_to_bf8_from_f32<stochastic_rounding>(float(a), rng);
    }
    return val;
#else // non gfx940
    return T(float(a),
             stochastic_rounding ? T::rocwmma_hip_f8_rounding_mode::stochastic
                                 : T::rocwmma_hip_f8_rounding_mode::standard,
             rng);
#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942
}

// NOTE NOTE: The above code is good if we don't consider HIP-GEMM code and only consider the quantization
// However, if we need HIP-GEMM for fall-back, we would need explicit_cast handles Tacc=f32 to To=f16/bf16 conversion
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && !(std::is_same<T, rocwmma_f8>{} || std::is_same<T, rocwmma_bf8>{})),
                            int>::type
    = 0>
inline ROCWMMA_HOST_DEVICE T explicit_downcast(Ta a, uint32_t rng)
{
    // the return type is not a F8 types, no SR for those types
    // not sure if we have direct conversion, so converting to float first
    // no effect if the input type is float
    return T(float(a));
}

// =================================================================================================

#endif // ROCWMMA_FLOAT8_H
