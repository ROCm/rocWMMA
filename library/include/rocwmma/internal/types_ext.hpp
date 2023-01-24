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

} // namespace rocwmma

#endif // ROCWMMA_TYPES_EXT_HPP
