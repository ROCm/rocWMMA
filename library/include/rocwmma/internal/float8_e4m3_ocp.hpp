/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_FLOAT8_E4M3_OCP_HPP
#define ROCWMMA_FLOAT8_E4M3_OCP_HPP

#include "float_conversion.hpp"

/// Defines the Float8_e4m3fn type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// bias = 7
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf
/// and inspired by Half implementation from pytorch/c10/util/Half.h

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_function__
#endif

namespace rocwmma
{

    namespace detail
    {

        /*
 * Convert a 8-bit floating-point number in fp8 E4M3FN format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
        inline ROCWMMA_HOST_DEVICE float fp8e4m3fn_to_fp32_value(uint8_t input)
        {
            /*
   * Extend the fp8 E4M3FN number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEE|MMM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 27-30 24-26          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
            const uint32_t w = (uint32_t)input << 24;
            /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
            const uint32_t sign = w & UINT32_C(0x80000000);
            /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+----+---+-----------------------------+
   *      | S |EEEE|MMM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31  27-30 24-26      0-23
   */
            const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
            /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 5 bits (sign == 0 and 4-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            uint32_t renorm_shift = __clz(nonsign);
#else
            // Note: zero is not a supported input into `__builtin_clz`
            uint32_t renorm_shift
                = nonsign != 0 ? __builtin_clz(nonsign) : sizeof(uint32_t) * CHAR_BIT;
#endif
            renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

            /*
   * Iff fp8e4m3fn number has all exponent and mantissa bits set to 1,
   * the addition overflows it into bit 31, and the subsequent shift turns the
   * high 9 bits into 1. Thus inf_nan_mask == 0x7F800000 if the fp8e4m3fn number
   * is Nan, 0x00000000 otherwise
   */
            const int32_t inf_nan_mask
                = ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
            /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
   * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
   * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
   * 0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
   * 0x00000000 otherwise
   */
            const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
            /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input
   * was denormal)
   * 2. Shift nonsign right by 4 so the exponent (4 bits originally)
   * becomes an 8-bit field and 3-bit mantissa shifts into the 3 high
   * bits of the 23-bit mantissa of IEEE single-precision number.
   * 3. Add 0x78 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0x07
   * for fp8e4m3fn number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to
   * account for renormalization. As renorm_shift is less than 0x78, this
   * can be combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the
   * input was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent
   * into zero if the input was zero.
   * 7. Combine with the sign of the input number.
   */
            uint32_t result = sign
                              | ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23))
                                  | inf_nan_mask)
                                 & ~zero_mask);
            return fp32_from_bits(result);
        }

        /*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FN format, in bit representation.
 */
        inline ROCWMMA_HOST_DEVICE uint8_t fp8e4m3fn_from_fp32_value(float f)
        {
            /*
   * Binary representation of 480.0f, which is the first value
   * not representable in fp8e4m3fn range:
   * 0 1111 111 - fp8e4m3fn
   * 0 10000111 11100000000000000000000 - fp32
   */
            constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

            /*
   * A mask for converting fp32 numbers lower than fp8e4m3fn normal range
   * into denorm representation
   * magic number: ((127 - 7) + (23 - 3) + 1)
   */
            constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

            uint32_t f_bits = fp32_to_bits(f);

            uint8_t result = 0u;

            /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
            const uint32_t sign = f_bits & UINT32_C(0x80000000);

            /*
   * Set sign bit to 0
   */
            f_bits ^= sign;

            if(f_bits >= fp8_max)
            {
                // NaN - all exponent and mantissa bits set to 1
                result = 0x7f;
            }
            else
            {
                if(f_bits < (UINT32_C(121) << 23))
                {
                    // Input number is smaller than 2^(-6), which is the smallest
                    // fp8e4m3fn normal number
                    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
                    result = static_cast<uint8_t>(f_bits - denorm_mask);
                }
                else
                {
                    // resulting mantissa is odd
                    uint8_t mant_odd = (f_bits >> 20) & 1;

                    // update exponent, rounding bias part 1
                    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

                    // rounding bias part 2
                    f_bits += mant_odd;

                    // take the bits!
                    result = static_cast<uint8_t>(f_bits >> 20);
                }
            }

            result |= static_cast<uint8_t>(sign >> 24);
            return result;
        }

    } // namespace detail

    struct alignas(1) Float8_e4m3fn
    {
        uint8_t x;

        struct from_bits_t
        {
        };
        ROCWMMA_HOST_DEVICE static constexpr from_bits_t from_bits()
        {
            return from_bits_t();
        }

        Float8_e4m3fn() = default;

        constexpr ROCWMMA_HOST_DEVICE Float8_e4m3fn(uint8_t bits, from_bits_t)
            : x(bits){};
        inline ROCWMMA_HOST_DEVICE      Float8_e4m3fn(float value);
        inline ROCWMMA_HOST_DEVICE      operator float() const;
        inline ROCWMMA_HOST_DEVICE bool isnan() const;
    };

    inline std::ostream& operator<<(std::ostream& out, const Float8_e4m3fn& value)
    {
        out << (float)value;
        return out;
    }

} // namespace rocwmma

namespace rocwmma
{

    /// Constructors

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn::Float8_e4m3fn(float value)
        : x(detail::fp8e4m3fn_from_fp32_value(value))
    {
    }

    /// Implicit conversions

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn::operator float() const
    {
        return detail::fp8e4m3fn_to_fp32_value(x);
    }

    /// Special values helper

    inline ROCWMMA_HOST_DEVICE bool Float8_e4m3fn::isnan() const
    {
        return (x & 0b01111111) == 0b01111111;
    }

    /// Arithmetic

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator+(const Float8_e4m3fn& a,
                                                       const Float8_e4m3fn& b)
    {
        return static_cast<float>(a) + static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn& a,
                                                       const Float8_e4m3fn& b)
    {
        return static_cast<float>(a) - static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator*(const Float8_e4m3fn& a,
                                                       const Float8_e4m3fn& b)
    {
        return static_cast<float>(a) * static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator/(
        const Float8_e4m3fn& a, const Float8_e4m3fn& b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<float>(a) / static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(const Float8_e4m3fn& a)
    {
        return -static_cast<float>(a);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn& operator+=(Float8_e4m3fn& a, const Float8_e4m3fn& b)
    {
        a = a + b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn& operator-=(Float8_e4m3fn& a, const Float8_e4m3fn& b)
    {
        a = a - b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn& operator*=(Float8_e4m3fn& a, const Float8_e4m3fn& b)
    {
        a = a * b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn& operator/=(Float8_e4m3fn& a, const Float8_e4m3fn& b)
    {
        a = a / b;
        return a;
    }

    /// Arithmetic with floats

    inline ROCWMMA_HOST_DEVICE float operator+(Float8_e4m3fn a, float b)
    {
        return static_cast<float>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE float operator-(Float8_e4m3fn a, float b)
    {
        return static_cast<float>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE float operator*(Float8_e4m3fn a, float b)
    {
        return static_cast<float>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE float operator/(Float8_e4m3fn a,
                                               float b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<float>(a) / b;
    }

    inline ROCWMMA_HOST_DEVICE float operator+(float a, Float8_e4m3fn b)
    {
        return a + static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float operator-(float a, Float8_e4m3fn b)
    {
        return a - static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float operator*(float a, Float8_e4m3fn b)
    {
        return a * static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float
        operator/(float a, Float8_e4m3fn b) __ubsan_ignore_float_divide_by_zero__
    {
        return a / static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE float& operator+=(float& a, const Float8_e4m3fn& b)
    {
        return a += static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator-=(float& a, const Float8_e4m3fn& b)
    {
        return a -= static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator*=(float& a, const Float8_e4m3fn& b)
    {
        return a *= static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator/=(float& a, const Float8_e4m3fn& b)
    {
        return a /= static_cast<float>(b);
    }

    /// Arithmetic with doubles

    inline ROCWMMA_HOST_DEVICE double operator+(Float8_e4m3fn a, double b)
    {
        return static_cast<double>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE double operator-(Float8_e4m3fn a, double b)
    {
        return static_cast<double>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE double operator*(Float8_e4m3fn a, double b)
    {
        return static_cast<double>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE double operator/(Float8_e4m3fn a,
                                                double b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<double>(a) / b;
    }

    inline ROCWMMA_HOST_DEVICE double operator+(double a, Float8_e4m3fn b)
    {
        return a + static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double operator-(double a, Float8_e4m3fn b)
    {
        return a - static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double operator*(double a, Float8_e4m3fn b)
    {
        return a * static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double
        operator/(double a, Float8_e4m3fn b) __ubsan_ignore_float_divide_by_zero__
    {
        return a / static_cast<double>(b);
    }

    /// Arithmetic with ints

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int b)
    {
        return a + static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int b)
    {
        return a - static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int b)
    {
        return a * static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int b)
    {
        return a / static_cast<Float8_e4m3fn>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator+(int a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(int a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator*(int a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator/(int a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) / b;
    }

    //// Arithmetic with int64_t

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator+(Float8_e4m3fn a, int64_t b)
    {
        return a + static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(Float8_e4m3fn a, int64_t b)
    {
        return a - static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator*(Float8_e4m3fn a, int64_t b)
    {
        return a * static_cast<Float8_e4m3fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator/(Float8_e4m3fn a, int64_t b)
    {
        return a / static_cast<Float8_e4m3fn>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator+(int64_t a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator-(int64_t a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator*(int64_t a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e4m3fn operator/(int64_t a, Float8_e4m3fn b)
    {
        return static_cast<Float8_e4m3fn>(a) / b;
    }

    /// NOTE: we do not define comparisons directly and instead rely on the implicit
    /// conversion from rocwmma::Float8_e4m3fn to float.

} // namespace rocwmma

namespace std
{

    template <>
    class numeric_limits<rocwmma::Float8_e4m3fn>
    {
    public:
        static constexpr bool is_specialized    = true;
        static constexpr bool is_signed         = true;
        static constexpr bool is_integer        = false;
        static constexpr bool is_exact          = false;
        static constexpr bool has_infinity      = false;
        static constexpr bool has_quiet_NaN     = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr auto has_denorm        = true;
        static constexpr auto has_denorm_loss   = true;
        static constexpr auto round_style       = numeric_limits<float>::round_style;
        static constexpr bool is_iec559         = false;
        static constexpr bool is_bounded        = true;
        static constexpr bool is_modulo         = false;
        static constexpr int  digits            = 4;
        static constexpr int  digits10          = 0;
        static constexpr int  max_digits10      = 3;
        static constexpr int  radix             = 2;
        static constexpr int  min_exponent      = -5;
        static constexpr int  min_exponent10    = -1;
        static constexpr int  max_exponent      = 8;
        static constexpr int  max_exponent10    = 2;
        static constexpr auto traps             = numeric_limits<float>::traps;
        static constexpr auto tinyness_before   = false;

        static constexpr rocwmma::Float8_e4m3fn min()
        {
            return rocwmma::Float8_e4m3fn(0x08, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn lowest()
        {
            return rocwmma::Float8_e4m3fn(0xFE, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn max()
        {
            return rocwmma::Float8_e4m3fn(0x7E, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn epsilon()
        {
            return rocwmma::Float8_e4m3fn(0x20, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn round_error()
        {
            return rocwmma::Float8_e4m3fn(0x30, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn quiet_NaN()
        {
            return rocwmma::Float8_e4m3fn(0x7F, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn signaling_NaN()
        {
            return rocwmma::Float8_e4m3fn(0x7F, rocwmma::Float8_e4m3fn::from_bits());
        }
        static constexpr rocwmma::Float8_e4m3fn denorm_min()
        {
            return rocwmma::Float8_e4m3fn(0x01, rocwmma::Float8_e4m3fn::from_bits());
        }
    };

} // namespace std

#endif // ROCWMMA_FLOAT8_E4M3_OCP_HPP
