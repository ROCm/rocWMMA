/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2023 Advanced Micro Devices, Inc.
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
        using BoolVecT = uint32_t __attribute__((ext_vector_type(next_pow2(Rank))));
        using VecT     = non_native_vector_base<T, Rank>;

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
        inline VecT& operator+=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline VecT& operator-=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline VecT& operator*=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline VecT& operator/=(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline VecT operator+(const VecT& x_) noexcept;

        ROCWMMA_HOST_DEVICE
        inline VecT operator-(const VecT& x_) noexcept;

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

// HIP doesn't have functional support for non-native vector types __half or bfloat16_t.
// Implement full support for those here.
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

#if defined(__gfx940__)

// HIP doesn't have functional support for non-native vector types bfloat8_t or float8_t.
// Implement full support for those here.
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::bfloat8_t, 512);

ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 1);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 2);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 3);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 4);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 8);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 16);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 32);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 64);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 128);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 256);
ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(rocwmma::float8_t, 512);

#endif // defined(__gfx940__)

// HIP bfloat16_t is not supported in RTC environment
#if !defined(__HIPCC_RTC__)

/// Register bfloat16_t vector types
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(1);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(2);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(3);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(4);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(8);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(16);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(32);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(64);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(128);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(256);
ROCWMMA_REGISTER_HIP_BFLOAT16_VECTOR_TYPE(512);

#endif // !__HIPCC_RTC__

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

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_HPP
