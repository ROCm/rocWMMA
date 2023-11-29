/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_VECTOR_HPP
#define ROCWMMA_VECTOR_HPP

// #include "types.hpp"
// #include "types_ext.hpp"
#if !defined(__HIPCC_RTC__)
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#endif

/**
 * rocWMMA vectors are implemented as HIP_vector_type<T, N> objects, which will ultimately
 * serve as the backend storage for fragment objects. The intention is to be compatible
 * with HIP vector types such that rocWMMA vectors:
 * - are interchangeable with other HIP vectors
 * - assume HIP's existing compatibility with nvcuda vectors
 * - are compatible with HIP RunTime Compilation (RTC) environment
 * - inherit performance and alignment benefits of HIP vectors
 *
 * At the time of this writing, the HIP_vector_type currently implements
 * vector sizes of 1, 2, 3 and 4 for native built-in datatypes. These vector classes may
 * be familiar to you as float1, float2, float3, ... types and so on.
 *
 * rocWMMA requires additional support for non-native datatypes (classes or structs implementing
 * custom data types such as __half and hip_bfloat16 that are not natively part of the platform),
 * as well as additional larger vector sizes. Thus rocWMMA will implement extensions to HIP vector
 * structures to accomodate these needs.
 *
 * HIP_vector_type class is comprised of 3 levels of interfaces (hip/amd_hip_vector_types.h):
 * 1. Native_vec_
 * 2. HIP_vector_base<T, N>
 * 3. HIP_vector_type<T, N>
 *
 * LEVEL 1: Native_vec_ (hip/amd_hip_vector_types.h)
 * These are the basic vector containers that are used as storage implementations.
 * Vector extensions are used by default if enabled by the compiler, otherwise it encapsulates
 * a built-in array.
 *
 * NOTE: Vector extensions are ONLY used on types native to the platform (e.g. char, int, float),
 * and IF the compiler supports the extensions.
 *
 * The interface of Native_vec_ objects MUST implement:
 * - Default CTOR, DTOR
 * - Copy CTOR
 * - Move CTOR
 * - Initializer CTOR
 * - Assignment Operators
 * - Element-wise access operators[int]
 * - Self assignment arithmetic operators (+=, -=, *=, /=, %=)
 * - Self assignment bitwise operators (|=, &=, ^=, >>=, <<=)
 * - Unary arithmetic and bitwise operators (-, ~)
 *
 * LEVEL 2: HIP_vector_base<T, N> (hip/amd_hip_vector_types.h)
 * This class encapsulates a Native_vec_ object as local storage, and implements element-wise
 * aliasing as the access interface. It is essentially a union of a vector object with named
 * accessor members x, y, z, w that alias respective numbered elements 0, 1, 2, 3 as the vector
 * size allows.
 * NOTE: Only explicit specialization for each size of vector (1, 2, 3, 4) has been implemented for this
 * class to control availability of named aliasing.
 *
 * LEVEL 3: HIP_vector_type<T, N> (hip/amd_hip_vector_types.h)
 * This class is a generic wrapper which inherits HIP_vector_base<T, N> and implements a more complete
 * generalized interface that includes all Level 1 interface, in addition to more bitwise, binary,
 * unary and relational public operators applicable to vectors.
 *
 * rocWMMA Extensions:
 * LEVEL 1: generalized non_native_vector_base<T, N> (vector.hpp, vector_impl.hpp)
 * This class is to be used as Native_vec_ when vector extensions are not supported, or non-native
 * datatypes are used. This class is a generalization of the LEVEL 1 interface for any type and any
 * rank (vector size). This allows rocWMMA to support non-native datatypes with HIP_vector_class.
 *
 * LEVEL 2: HIP_vector_base<T, N> registration (vector_impl.hpp):
 * ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, STORAGE_IMPL)
 * This macro implements HIP_vector_base specializations for any TYPE and RANK, and for either native or
 * non-native data type storage. This allows rocWMMA to support larger RANKs of vectors beyond what is
 * already supported.
 *
 * LEVEL 3: HIP_vector_type<T, N> registration (vector_impl.hpp):
 * ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(TYPE, RANK) and ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(RANK)
 * These macros implement specific specializations for supporting either native or non-native datatypes
 * in the HIP_vector_type<T, N> interface.
 *
 */

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}
namespace rocwmma
{
    template <typename T, unsigned int Rank>
    struct non_native_vector_base
    {
        /// Types
        using BoolVecT = non_native_vector_base<bool, Rank>;
        using VecT     = non_native_vector_base<T, Rank>;

        ROCWMMA_HOST_DEVICE
        constexpr static inline uint32_t size()
        {
            return Rank;
        }

        /// Ctor, dtor, assignment
        ROCWMMA_HOST_DEVICE
        non_native_vector_base() = default;

        ROCWMMA_HOST_DEVICE
        constexpr non_native_vector_base(const VecT&) = default;

        ROCWMMA_HOST_DEVICE
        constexpr non_native_vector_base(VecT&&) = default;

        ROCWMMA_HOST_DEVICE
        ~non_native_vector_base() = default;

        ROCWMMA_HOST_DEVICE
        inline VecT& operator=(const VecT&) = default;

        ROCWMMA_HOST_DEVICE
        inline VecT& operator=(VecT&&) = default;

        template <typename U                                                           = T,
                  typename std::enable_if<(std::is_same<U, T>{}) && (Rank > 1)>::type* = nullptr>
        ROCWMMA_HOST_DEVICE explicit constexpr non_native_vector_base(T x_) noexcept;

        template <typename... Ts,
                  typename U                                              = T,
                  typename std::enable_if<(sizeof...(Ts) == Rank)>::type* = nullptr>
        ROCWMMA_HOST_DEVICE constexpr non_native_vector_base(Ts... args) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline T& operator[](unsigned int idx) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline T operator[](unsigned int idx) const noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT& operator+=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT& operator-=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT& operator*=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT& operator/=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT operator+(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT operator-(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT operator*(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        constexpr inline VecT operator/(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator%=(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT operator-() const noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator&=(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator|=(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT operator~() const noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator^=(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator>>=(const VecT& x_) noexcept;

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        ROCWMMA_HOST_DEVICE inline VecT& operator<<=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator==(const VecT& x_) const noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator!=(const VecT& x_) const noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator>=(const VecT& x_) const noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator<=(const VecT& x_) const noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator>(const VecT& x_) const noexcept;

        ROCWMMA_HOST_DEVICE
        inline BoolVecT operator<(const VecT& x_) const noexcept;

        /// Storage
        T d[Rank];
    };

} // namespace rocwmma

#include "vector_impl.hpp"

// Implements HIP_vector_type for rocWMMA types.
// HIP already has built-in support for native data type vectors of size <= 4.
// Implement support for powers of 2 up to and including 256.
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float16_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float32_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::float64_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int8_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint8_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int16_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint16_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int32_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint32_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::int64_t, 512);

ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 8);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 16);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 32);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 64);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 128);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 256);
ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(rocwmma::uint64_t, 512);

// HIP doesn't have functional support for non-native vector types __half or bfloat16_t.
// Implement full support for those here.
#if !ROCWMMA_NO_HALF
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::hfloat16_t, 512);
#endif // !ROCWMMA_NO_HALF

// Register bfloat8_t vector types
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat8_t, 512);

// Register float8_t vector types
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::float8_t, 512);

ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::xfloat32_t, 512);

/// Register bfloat16_t vector types
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(rocwmma::bfloat16_t, 512);

namespace rocwmma
{
    template <typename VecT>
    struct VecTraits;

    template <typename T, uint32_t VecSize>
    struct VecTraits<HIP_vector_type<T, VecSize>>
    {
        // Vector class blueprint
        template <typename DataT = T, uint32_t size = VecSize>
        using VecT = HIP_vector_type<T, size>;

        // Current data type
        using DataT = typename VecT<>::value_type;

        // Current vector size
        constexpr static inline uint32_t size()
        {
            return VecSize;
        }
    };

    template <typename T, uint32_t VecSize>
    struct VecTraits<non_native_vector_base<T, VecSize>>
    {
        // Vector class blueprint
        template <typename DataT = T, uint32_t size = VecSize>
        using VecT = non_native_vector_base<T, size>;

        // Current data type
        using DataT = T;

        // Current vector size
        constexpr static inline uint32_t size()
        {
            return VecSize;
        }
    };

    namespace detail
    {
        template <typename T, typename... Ts>
        constexpr auto getFirstType()
        {
            return T();
        }

        template <typename T>
        constexpr bool isSameType()
        {
            return true;
        }

        template <typename T, typename Y, typename... Ts>
        constexpr bool isSameType()
        {
            return std::is_same<T, Y>::value && isSameType<Y, Ts...>();
        }
    }

    ///////////////////////////////////////////////////////////////////
    ///           HIP_vector_type<T, N> utility overrides           ///
    ///                                                             ///
    /// Note: HIP_vector_type<T, N> uses vector extensions.         ///
    /// Element-wise access of vectors in constexpr is forbidden.   ///
    ///////////////////////////////////////////////////////////////////
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr inline DataT& get(HIP_vector_type<DataT, VecSize>& v)
    {
        return reinterpret_cast<DataT*>(&v.data)[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr inline DataT get(HIP_vector_type<DataT, VecSize> const& v)
    {
        return v.data[Idx];
    }

    template <typename DataT>
    ROCWMMA_HOST_DEVICE constexpr inline auto swap(HIP_vector_type<DataT, 2> const& v)
    {
        return HIP_vector_type<DataT, 2>{get<1>(v), get<0>(v)};
    }

    namespace detail
    {
        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        constexpr decltype(auto)
            apply_impl(F fn, HIP_vector_type<DataT, Rank> const& v, index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }

    } // namespace detail

    template <typename F, typename DataT, uint32_t Rank>
    constexpr decltype(auto) apply(F fn, HIP_vector_type<DataT, Rank>& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    ///////////////////////////////////////////////////////////////////
    ///     non_native_vector_base<T, N> utility overrides          ///
    ///////////////////////////////////////////////////////////////////
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline DataT&
        get(non_native_vector_base<DataT, VecSize>& v)
    {
        return v[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline DataT
        get(non_native_vector_base<DataT, VecSize> const& v)
    {
        return v[Idx];
    }

    namespace detail
    {
        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        constexpr decltype(auto)
            apply_impl(F fn, non_native_vector_base<DataT, Rank> const& v, index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }

    } // namespace detail

    template <typename F, typename DataT, uint32_t Rank>
    constexpr decltype(auto) apply(F fn, non_native_vector_base<DataT, Rank> const& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename... Ts>
    constexpr decltype(auto) make_vector(Ts&&... ts)
    {
        // TODO: When HIP_vector_type becomes constexpr replace with non_native_vector type.

        // Ensure that all the arguments are the same type
        static_assert(detail::isSameType<std::decay_t<Ts>...>(),
                      "Vector arguments must all be the same type");

        using DataT = decltype(detail::getFirstType<std::decay_t<Ts>...>());
        return non_native_vector_base<DataT, sizeof...(Ts)>{std::forward<Ts>(ts)...};
    }

    namespace detail
    {
        template <typename DataT0,
                  uint32_t Rank0,
                  size_t... Is0,
                  typename DataT1,
                  uint32_t Rank1,
                  size_t... Is1>
        constexpr static inline decltype(auto)
            vector_cat_impl(non_native_vector_base<DataT0, Rank0> const& lhs,
                            index_sequence<Is0...>,
                            non_native_vector_base<DataT1, Rank1> const& rhs,
                            index_sequence<Is1...>)
        {
            return make_vector(get<Is0>(lhs)..., get<Is1>(rhs)...);
        }

    } // namespace detail

    template <typename Lhs, typename Rhs>
    constexpr decltype(auto) vector_cat(Lhs&& lhs, Rhs&& rhs)
    {
        constexpr std::size_t Size0 = VecTraits<std::decay_t<decltype(lhs)>>::size();
        constexpr std::size_t Size1 = VecTraits<std::decay_t<decltype(rhs)>>::size();

        return detail::vector_cat_impl(std::forward<Lhs>(lhs),
                                       detail::make_index_sequence<Size0>(),
                                       std::forward<Rhs>(rhs),
                                       detail::make_index_sequence<Size1>());
    }

    namespace detail
    {
        template <typename DataT0, typename DataT1, uint32_t Rank, size_t... Is>
        constexpr static inline decltype(auto)
            mult_poly_vec_impl(non_native_vector_base<DataT0, Rank> const& lhs,
                               non_native_vector_base<DataT1, Rank> const& rhs,
                               index_sequence<Is...>)
        {
            return make_vector((get<Is>(lhs) * get<Is>(rhs))...);
        }

    } // namespace detail

    template <typename DataT0, typename DataT1, uint32_t Rank>
    constexpr decltype(auto) operator*(non_native_vector_base<DataT0, Rank> const& lhs,
                                       non_native_vector_base<DataT1, Rank> const& rhs)
    {
        return detail::mult_poly_vec_impl(lhs, rhs, detail::make_index_sequence<Rank>());
    }

    namespace detail
    {
        template <class BinOp, typename T, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline std::decay_t<T>
            reduceOp_impl(T&& t, Ts&&... ts) noexcept
        {
            using CastT = std::decay_t<T>;
            if constexpr(sizeof...(Ts) >= 1)
            {
                return BinOp::exec(static_cast<CastT>(t),
                                   reduceOp_impl<BinOp>(std::forward<Ts>(ts)...));
            }
            else
            {
                return static_cast<CastT>(t);
            }
        }

        template <class BinOp, typename VecT, size_t... Is>
        ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
            vector_reduce_impl(VecT&& v, index_sequence<Is...>) noexcept
        {
            return reduceOp_impl<BinOp>(get<Is>(v)...);
        }

        // Use with operations that have 1 operands
        template <class BinOp, typename VecT>
        ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
            vector_reduce(VecT&& lhs) noexcept
        {
            return vector_reduce_impl<BinOp>(
                std::forward<VecT>(lhs),
                detail::make_index_sequence<VecTraits<std::decay_t<VecT>>::size()>{});
        }
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
        vector_reduce_and(VecT&& lhs) noexcept
    {
        return detail::vector_reduce<detail::BitwiseOp::And>(std::forward<VecT>(lhs));
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_HPP
