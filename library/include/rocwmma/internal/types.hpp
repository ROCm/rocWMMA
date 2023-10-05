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

// Continue with vector type definitions
#include "vector.hpp"

namespace rocwmma
{

    /*! \class VecT
    *  \brief  HIP vector class
    *  @tparam DataT vector data type
    *  @tparam Rank vector size
    */
    template <typename DataT, uint32_t Rank>
    using VecT = HIP_vector_type<DataT, Rank>;

    // MFMA vector registers
    using VRegI8x1  = VecT<int8_t, 1>; // Single i8 register
    using VRegI8x2  = VecT<int8_t, 2>; // Two i8 registers
    using VRegI8x4  = VecT<int8_t, 4>; // ...
    using VRegI8x8  = VecT<int8_t, 8>; //
    using VRegI8x16 = VecT<int8_t, 16>; //
    using VRegI8x32 = VecT<int8_t, 32>; // 32 i8 registers

    using VRegI32x1  = VecT<int32_t, 1>; // Single i32 register
    using VRegI32x2  = VecT<int32_t, 2>; // Two i32 registers
    using VRegI32x4  = VecT<int32_t, 4>; // ...
    using VRegI32x8  = VecT<int32_t, 8>; //
    using VRegI32x16 = VecT<int32_t, 16>; //
    using VRegI32x32 = VecT<int32_t, 32>; // 32 i32 registers

    using VRegI64x1  = VecT<int64_t, 1>; // Single i64 register
    using VRegI64x2  = VecT<int64_t, 2>; // Two i64 registers
    using VRegI64x4  = VecT<int64_t, 4>; // ...
    using VRegI64x8  = VecT<int64_t, 8>; //
    using VRegI64x16 = VecT<int64_t, 16>; //
    using VRegI64x32 = VecT<int64_t, 32>; // 32 i64 registers

    using VRegF16x1  = VecT<float16_t, 1>; // Single f16 register
    using VRegF16x2  = VecT<float16_t, 2>; // Two f16 registers
    using VRegF16x4  = VecT<float16_t, 4>; // ...
    using VRegF16x8  = VecT<float16_t, 8>; //
    using VRegF16x16 = VecT<float16_t, 16>; //
    using VRegF16x32 = VecT<float16_t, 32>; // 32 f16 registers

    using VRegF32x1  = VecT<float32_t, 1>; // Single f32 register
    using VRegF32x2  = VecT<float32_t, 2>; // Two f32 registers
    using VRegF32x4  = VecT<float32_t, 4>; // ...
    using VRegF32x8  = VecT<float32_t, 8>; //
    using VRegF32x16 = VecT<float32_t, 16>; //
    using VRegF32x32 = VecT<float32_t, 32>; // 32 f32 registers

    using VRegF64x1  = VecT<float64_t, 1>; // Single f64 register
    using VRegF64x2  = VecT<float64_t, 2>; // Two f64 registers
    using VRegF64x4  = VecT<float64_t, 4>; // ...
    using VRegF64x8  = VecT<float64_t, 8>; //
    using VRegF64x16 = VecT<float64_t, 16>; //
    using VRegF64x32 = VecT<float64_t, 32>; // 32 f64 registers

    // Acc registers
    using AccRegI32x1  = VecT<int32_t, 1>;
    using AccRegI32x2  = VecT<int32_t, 2>;
    using AccRegI32x4  = VecT<int32_t, 4>;
    using AccRegI32x8  = VecT<int32_t, 8>;
    using AccRegI32x16 = VecT<int32_t, 16>;
    using AccRegI32x32 = VecT<int32_t, 32>;

    using AccRegF32x1  = VecT<float32_t, 1>;
    using AccRegF32x2  = VecT<float32_t, 2>;
    using AccRegF32x4  = VecT<float32_t, 4>;
    using AccRegF32x8  = VecT<float32_t, 8>;
    using AccRegF32x16 = VecT<float32_t, 16>;
    using AccRegF32x32 = VecT<float32_t, 32>;

    using AccRegF64x1  = VecT<float64_t, 1>;
    using AccRegF64x2  = VecT<float64_t, 2>;
    using AccRegF64x4  = VecT<float64_t, 4>;
    using AccRegF64x8  = VecT<float64_t, 8>;
    using AccRegF64x16 = VecT<float64_t, 16>;
    using AccRegF64x32 = VecT<float64_t, 32>;

    using Coord2dDataT = uint32_t;
    using Coord2d      = non_native_vector_base<Coord2dDataT, 2>;

} // namespace rocwmma

// Add in some extensions to basic type support.
// Some of these are required for vector implementations.
#include "type_traits.hpp"
#include "types_ext.hpp"

#include "types_impl.hpp"

#endif // ROCWMMA_TYPES_HPP
