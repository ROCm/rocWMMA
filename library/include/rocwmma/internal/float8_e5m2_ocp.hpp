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

#ifndef ROCWMMA_FLOAT8_E5M2_OCP_HPP
#define ROCWMMA_FLOAT8_E5M2_OCP_HPP

#include "float_conversion.hpp"

/// Defines the Float8_e5m2fn type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// bias = 15
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
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
        ROCWMMA_HOST_DEVICE inline float fp16_ieee_to_fp32_value(uint16_t h)
        {
            /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
            const uint32_t w = (uint32_t)h << 16;
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
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
            const uint32_t two_w = w + w;

            /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
            constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
            // const float exp_scale = 0x1.0p-112f;
            constexpr uint32_t scale_bits    = (uint32_t)15 << 23;
            float              exp_scale_val = 0;
            std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
            const float exp_scale        = exp_scale_val;
            const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

            /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructed single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
            constexpr uint32_t magic_mask = UINT32_C(126) << 23;
            constexpr float    magic_bias = 0.5f;
            const float        denormalized_value
                = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

            /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
            constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
            const uint32_t     result
                = sign
                  | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                                 : fp32_to_bits(normalized_value));
            return fp32_from_bits(result);
        }

        /*
 * Convert a 8-bit floating-point number in fp8 E5M2FN format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
        inline ROCWMMA_HOST_DEVICE float fp8e5m2fn_to_fp32_value(uint8_t input)
        {
            /*
   * Extend the fp8 E5M2FN number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+--+-----------------------------+
   *      | S |EEEEE|MM|0000 0000 0000 0000 0000 0000|
   *      +---+-----+--+-----------------------------+
   * Bits  31 26-30 24-25          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
            uint16_t half_representation = input;
            half_representation <<= 8;
            return fp16_ieee_to_fp32_value(half_representation);
        }

        /*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2FN format, in bit representation.
 */
        inline ROCWMMA_HOST_DEVICE uint8_t fp8e5m2fn_from_fp32_value(float f)
        {
            /*
   * Binary representation of fp32 infinity
   * 0 11111111 00000000000000000000000
   */
            constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

            /*
   * Binary representation of 65536.0f, which is the first value
   * not representable in fp8e5m2 range:
   * 0 11111 00 - fp8e5m2
   * 0 10001111 00000000000000000000000 - fp32
   */
            constexpr uint32_t fp8_max = UINT32_C(143) << 23;

            /*
   * A mask for converting fp32 numbers lower than fp8e5m2 normal range
   * into denorm representation
   * magic number: ((127 - 15) + (23 - 2) + 1)
   */
            constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

            uint32_t f_bits = fp32_to_bits(f);
            uint8_t  result = 0u;

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
                result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
            }
            else
            {
                if(f_bits < (UINT32_C(113) << 23))
                {
                    // Input number is smaller than 2^(-14), which is the smallest
                    // fp8e5m2 normal number
                    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
                    result = static_cast<uint8_t>(f_bits - denorm_mask);
                }
                else
                {
                    // resulting mantissa is odd
                    uint32_t mant_odd = (f_bits >> 21) & 1;

                    // update exponent, rounding bias part 1
                    f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

                    // rounding bias part 2
                    f_bits += mant_odd;

                    // take the bits!
                    result = static_cast<uint8_t>(f_bits >> 21);
                }
            }

            result |= static_cast<uint8_t>(sign >> 24);
            return result;
        }

    } // namespace detail

    struct alignas(1) Float8_e5m2fn
    {
        uint8_t x;

        struct from_bits_t
        {
        };
        ROCWMMA_HOST_DEVICE static constexpr from_bits_t from_bits()
        {
            return from_bits_t();
        }

        Float8_e5m2fn() = default;

        constexpr ROCWMMA_HOST_DEVICE Float8_e5m2fn(uint8_t bits, from_bits_t)
            : x(bits){};
        inline ROCWMMA_HOST_DEVICE      Float8_e5m2fn(float value);
        inline ROCWMMA_HOST_DEVICE      operator float() const;
        inline ROCWMMA_HOST_DEVICE bool isnan() const;
        inline ROCWMMA_HOST_DEVICE bool isinf() const;
    };

    inline std::ostream& operator<<(std::ostream& out, const Float8_e5m2fn& value)
    {
        out << (float)value;
        return out;
    }

} // namespace rocwmma

namespace rocwmma
{

    /// Constructors

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn::Float8_e5m2fn(float value)
        : x(detail::fp8e5m2fn_from_fp32_value(value))
    {
    }

    /// Implicit conversions

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn::operator float() const
    {
        return detail::fp8e5m2fn_to_fp32_value(x);
    }

    /// Special values helper

    inline ROCWMMA_HOST_DEVICE bool Float8_e5m2fn::isnan() const
    {
        return (x & 0b01111111) > 0b01111100;
    }

    inline ROCWMMA_HOST_DEVICE bool Float8_e5m2fn::isinf() const
    {
        return (x & 0b01111111) == 0b01111100;
    }

    /// Arithmetic

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator+(const Float8_e5m2fn& a,
                                                       const Float8_e5m2fn& b)
    {
        return static_cast<float>(a) + static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(const Float8_e5m2fn& a,
                                                       const Float8_e5m2fn& b)
    {
        return static_cast<float>(a) - static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator*(const Float8_e5m2fn& a,
                                                       const Float8_e5m2fn& b)
    {
        return static_cast<float>(a) * static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator/(
        const Float8_e5m2fn& a, const Float8_e5m2fn& b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<float>(a) / static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(const Float8_e5m2fn& a)
    {
        return -static_cast<float>(a);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn& operator+=(Float8_e5m2fn& a, const Float8_e5m2fn& b)
    {
        a = a + b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn& operator-=(Float8_e5m2fn& a, const Float8_e5m2fn& b)
    {
        a = a - b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn& operator*=(Float8_e5m2fn& a, const Float8_e5m2fn& b)
    {
        a = a * b;
        return a;
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn& operator/=(Float8_e5m2fn& a, const Float8_e5m2fn& b)
    {
        a = a / b;
        return a;
    }

    /// Arithmetic with floats

    inline ROCWMMA_HOST_DEVICE float operator+(Float8_e5m2fn a, float b)
    {
        return static_cast<float>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE float operator-(Float8_e5m2fn a, float b)
    {
        return static_cast<float>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE float operator*(Float8_e5m2fn a, float b)
    {
        return static_cast<float>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE float operator/(Float8_e5m2fn a,
                                               float b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<float>(a) / b;
    }

    inline ROCWMMA_HOST_DEVICE float operator+(float a, Float8_e5m2fn b)
    {
        return a + static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float operator-(float a, Float8_e5m2fn b)
    {
        return a - static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float operator*(float a, Float8_e5m2fn b)
    {
        return a * static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float
        operator/(float a, Float8_e5m2fn b) __ubsan_ignore_float_divide_by_zero__
    {
        return a / static_cast<float>(b);
    }

    inline ROCWMMA_HOST_DEVICE float& operator+=(float& a, const Float8_e5m2fn& b)
    {
        return a += static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator-=(float& a, const Float8_e5m2fn& b)
    {
        return a -= static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator*=(float& a, const Float8_e5m2fn& b)
    {
        return a *= static_cast<float>(b);
    }
    inline ROCWMMA_HOST_DEVICE float& operator/=(float& a, const Float8_e5m2fn& b)
    {
        return a /= static_cast<float>(b);
    }

    /// Arithmetic with doubles

    inline ROCWMMA_HOST_DEVICE double operator+(Float8_e5m2fn a, double b)
    {
        return static_cast<double>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE double operator-(Float8_e5m2fn a, double b)
    {
        return static_cast<double>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE double operator*(Float8_e5m2fn a, double b)
    {
        return static_cast<double>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE double operator/(Float8_e5m2fn a,
                                                double b) __ubsan_ignore_float_divide_by_zero__
    {
        return static_cast<double>(a) / b;
    }

    inline ROCWMMA_HOST_DEVICE double operator+(double a, Float8_e5m2fn b)
    {
        return a + static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double operator-(double a, Float8_e5m2fn b)
    {
        return a - static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double operator*(double a, Float8_e5m2fn b)
    {
        return a * static_cast<double>(b);
    }
    inline ROCWMMA_HOST_DEVICE double
        operator/(double a, Float8_e5m2fn b) __ubsan_ignore_float_divide_by_zero__
    {
        return a / static_cast<double>(b);
    }

    /// Arithmetic with ints

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator+(Float8_e5m2fn a, int b)
    {
        return a + static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(Float8_e5m2fn a, int b)
    {
        return a - static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator*(Float8_e5m2fn a, int b)
    {
        return a * static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator/(Float8_e5m2fn a, int b)
    {
        return a / static_cast<Float8_e5m2fn>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator+(int a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(int a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator*(int a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator/(int a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) / b;
    }

    //// Arithmetic with int64_t

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator+(Float8_e5m2fn a, int64_t b)
    {
        return a + static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(Float8_e5m2fn a, int64_t b)
    {
        return a - static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator*(Float8_e5m2fn a, int64_t b)
    {
        return a * static_cast<Float8_e5m2fn>(b);
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator/(Float8_e5m2fn a, int64_t b)
    {
        return a / static_cast<Float8_e5m2fn>(b);
    }

    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator+(int64_t a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) + b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator-(int64_t a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) - b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator*(int64_t a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) * b;
    }
    inline ROCWMMA_HOST_DEVICE Float8_e5m2fn operator/(int64_t a, Float8_e5m2fn b)
    {
        return static_cast<Float8_e5m2fn>(a) / b;
    }

    /// NOTE: we do not define comparisons directly and instead rely on the implicit
    /// conversion from rocwmma::Float8_e5m2fn to float.

} // namespace rocwmma

namespace std
{

    template <>
    class numeric_limits<rocwmma::Float8_e5m2fn>
    {
    public:
        static constexpr bool is_signed         = true;
        static constexpr bool is_integer        = false;
        static constexpr bool is_specialized    = true;
        static constexpr bool is_exact          = false;
        static constexpr bool has_infinity      = true;
        static constexpr bool has_quiet_NaN     = false;
        static constexpr bool has_signaling_NaN = false;
        static constexpr auto has_denorm        = true;
        static constexpr auto has_denorm_loss   = true;
        static constexpr auto round_style       = numeric_limits<float>::round_style;
        static constexpr bool is_iec559         = false;
        static constexpr bool is_bounded        = true;
        static constexpr bool is_modulo         = false;
        static constexpr int  digits            = 3;
        static constexpr int  digits10          = 0;
        static constexpr int  max_digits10      = 2;
        static constexpr int  radix             = 2;
        static constexpr int  min_exponent      = -13;
        static constexpr int  min_exponent10    = -4;
        static constexpr int  max_exponent      = 16;
        static constexpr int  max_exponent10    = 4;
        static constexpr auto traps             = numeric_limits<float>::traps;
        static constexpr auto tinyness_before   = false;

        static constexpr rocwmma::Float8_e5m2fn min()
        {
            return rocwmma::Float8_e5m2fn(0x04, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn max()
        {
            return rocwmma::Float8_e5m2fn(0x7B, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn lowest()
        {
            return rocwmma::Float8_e5m2fn(0xFB, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn epsilon()
        {
            return rocwmma::Float8_e5m2fn(0x34, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn round_error()
        {
            return rocwmma::Float8_e5m2fn(0x38, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn infinity()
        {
            return rocwmma::Float8_e5m2fn(0x7C, rocwmma::Float8_e5m2fn::from_bits());
        }
        static constexpr rocwmma::Float8_e5m2fn denorm_min()
        {
            return rocwmma::Float8_e5m2fn(0x01, rocwmma::Float8_e5m2fn::from_bits());
        }
    };

} // namespace std

#endif // ROCWMMA_FLOAT8_E5M2_OCP_HPP
