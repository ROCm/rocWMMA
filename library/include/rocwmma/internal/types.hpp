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

#if ROCWMMA_ENABLE_FP8_OCP
#include "float8.hpp"
#include "float8_e4m3_ocp.hpp"
#include "float8_e5m2_ocp.hpp"
#elif ROCWMMA_ENABLE_FP8_NANOO
#include "float8.hpp"
#endif

#include "rocwmma_xfloat32.hpp"

namespace rocwmma
{

    /**
 * \defgroup Datatypes Datatypes
 *
 * @brief Summary of datatypes used in rocWMMA.
 *
 * @{
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

    using float8_t = std::conditional_t<(bool)ROCWMMA_ENABLE_FP8_OCP, Float8_e4m3fn, rocwmma_f8>;

    using bfloat8_t = std::conditional_t<(bool)ROCWMMA_ENABLE_FP8_OCP, Float8_e5m2fn, rocwmma_bf8>;

    using xfloat32_t = rocwmma_xfloat32;

    /** @}*/

} // namespace rocwmma

// Add in some extensions to basic type support.
// Some of these are required for vector implementations.
#include "type_traits.hpp"
#include "types_ext.hpp"

#include "types_impl.hpp"

#endif // ROCWMMA_TYPES_HPP
