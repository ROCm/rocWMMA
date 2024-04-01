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
#ifndef ROCWMMA_TYPES_HPP
#define ROCWMMA_TYPES_HPP

#if !defined(__HIPCC_RTC__)
#include <array>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#include <type_traits>
#include <utility>
#endif // !__HIPCC_RTC__

#include "config.hpp"
#include "float8.h"
#include "rocwmma_xfloat32.hpp"

namespace rocwmma
{

    /**
 * \defgroup DataTypes Data Type Metadata
 *
 * @brief Definition and metadata on supported data types of matrices.
 *
 * @{
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

    // Native types
    using float16_t = _Float16;
    using float32_t = float;
    using float64_t = double;

#if !defined(__HIPCC_RTC__)

    using int8_t   = ::int8_t;
    using uint8_t  = ::uint8_t;
    using int16_t  = ::int16_t;
    using uint16_t = ::uint16_t;
    using int32_t  = ::int32_t;
    using uint32_t = ::uint32_t;
    using int64_t  = ::int64_t;
    using uint64_t = ::uint64_t;
    using index_t  = ::int32_t;

#else

    using int8_t   = __hip_internal::int8_t;
    using uint8_t  = __hip_internal::uint8_t;
    using int16_t  = __hip_internal::int16_t;
    using uint16_t = __hip_internal::uint16_t;
    using int32_t  = __hip_internal::int32_t;
    using uint32_t = __hip_internal::uint32_t;
    using int64_t  = __hip_internal::int64_t;
    using uint64_t = __hip_internal::uint64_t;
    using index_t  = __hip_internal::int32_t;

#endif // !defined(__HIPCC_RTC__)

    // Non-native types
    using bfloat16_t = hip_bfloat16;

#if !ROCWMMA_NO_HALF
    using hfloat16_t = __half;
#endif // !ROCWMMA_NO_HALF

    using bfloat8_t = rocwmma_bf8;
    using float8_t  = rocwmma_f8;

    using xfloat32_t = rocwmma_xfloat32;

    // clang-format off

    // Data layout meta-tags
    /*! \struct row_major
 *  \brief Data/In-memory Layout as Row Major
 */
    struct row_major{};
    /*! \struct col_major
 *  \brief Data/In-memory Layout as Column Major
 */
    struct col_major{};

    // Fragment usage meta-tags
    /*! \struct matrix_a
 *  \brief Input Matrix A
 */
    struct matrix_a{};
    /*! \struct matrix_b
 *  \brief Input Matrix B
 */
    struct matrix_b{};
    /*! \struct accumulator
 *  \brief Input/Output Matrix Accumulator
 */
    struct accumulator{};

    // clang-format on

    /*! \struct layout_t
 *  \brief Definition of Runtime data layout flags
 *  @var mem_row_major
 *  @var mem_col_major
 */
    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };
    /** @}*/

} // namespace rocwmma

// Add in some extensions to basic type support.
// Some of these are required for vector implementations.
#include "type_traits.hpp"
#include "types_ext.hpp"

#include "types_impl.hpp"

#endif // ROCWMMA_TYPES_HPP
