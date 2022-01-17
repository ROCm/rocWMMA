/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef WMMA_TYPES_H
#define WMMA_TYPES_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <type_traits>

/**
 * \ingroup wmma
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
// Native types
using float32_t = float;
using float16_t = _Float16;
using float64_t = double;
using int8_t    = signed char;
using uint8_t   = unsigned char;
using int16_t   = short;
using int32_t   = int;
using uint32_t  = unsigned int;
using index_t   = int32_t;

// Non-native types
using bfloat16_t = hip_bfloat16;
using hfloat16_t = __half;

namespace std
{
    inline ostream& operator<<(ostream& stream, float16_t const& val)
    {
        return stream << static_cast<float>(val);
    }

    inline ostream& operator<<(ostream& stream, hfloat16_t const& val)
    {
        return stream << __half2float(val);
    }
} // namespace std

// Vector internal storage
template <typename T, int Elements, typename IsNativeType = typename std::is_fundamental<T>::type>
struct VectorStorage;

// Native types can use explicit vector extension
template <typename T, int Elements>
struct VectorStorage<T, Elements, std::true_type>
{
    using type = T __attribute__((ext_vector_type(Elements)));
};

// Non-native types can use std::arrays.
// std::arrays has the same memory footprint as C-arrays
// but carry extra useful functionality.
// This allows us to use non-native data types as "vectors".
template <typename T, int Elements>
struct VectorStorage<T, Elements, std::false_type>
{
    using type = std::array<T, Elements>;
};

// Only vectorize for elements > 1
template <typename T, int Elements>
using _VecT =
    typename std::conditional<Elements == 1, T, typename VectorStorage<T, Elements>::type>::type;

// Vectors of f16
using v2_f16_t  = _VecT<float16_t, 2>;
using v4_f16_t  = _VecT<float16_t, 4>;
using v8_f16_t  = _VecT<float16_t, 8>;
using v16_f16_t = _VecT<float16_t, 16>;
using v32_f16_t = _VecT<float16_t, 32>;
using v64_f16_t = _VecT<float16_t, 64>;

// Vectors of f32
using v2_f32_t  = _VecT<float32_t, 2>;
using v4_f32_t  = _VecT<float32_t, 4>;
using v8_f32_t  = _VecT<float32_t, 8>;
using v16_f32_t = _VecT<float32_t, 16>;
using v32_f32_t = _VecT<float32_t, 32>;
using v64_f32_t = _VecT<float32_t, 64>;

// Vectors of ints
using v2_i32_t  = _VecT<int32_t, 2>;
using v4_i32_t  = _VecT<int32_t, 4>;
using v16_i32_t = _VecT<int32_t, 16>;
using v32_i32_t = _VecT<int32_t, 32>;
using v64_i32_t = _VecT<int32_t, 64>;

// Vector wrapper for element access.
template <typename T, uint32_t VecSize>
struct VecT;

template <typename T, uint32_t VecSize>
struct __align__(4) VecT
{
    using StorageT                 = _VecT<T, VecSize>;
    using DataT                    = T;
    constexpr static uint32_t Size = VecSize;

    union
    {
        static_assert(sizeof(StorageT) == sizeof(DataT[VecSize]),
                      "Unable to vectorize with StorageT");
        StorageT v; // Vector representation
        DataT    e[VecSize]; // Element array representation
    };

    __device__ VecT()  = default;
    __device__ ~VecT() = default;

    __device__ inline VecT(VecT const& other)
    {
        v = other.v;
    }

    __device__ inline VecT(StorageT const& other)
    {
        v = other;
    }

    __device__ VecT(StorageT && other)
    {
        v = std::move(other);
    }

    __device__ DataT& operator[](uint32_t index)
    {
        return e[index];
    }

    __device__ StorageT& operator*()
    {
        return v;
    }

    __device__ DataT const& operator[](uint32_t index) const
    {
        return e[index];
    }

    __device__ StorageT const& operator*() const
    {
        return v;
    }

    __device__ constexpr static inline uint32_t size()
    {
        return VecSize;
    }

    template <uint32_t SubVecSize>
    struct Iterator
    {
    private:
        int32_t               mIndex = 0;
        VecT<DataT, VecSize>& mParent;

    public:
        struct Traits
        {
            // Vector type pointed to by current iterator
            using ItVecT = _VecT<DataT, SubVecSize>;

            enum : int32_t
            {
                Range = VecSize / SubVecSize
            };
        };

        static_assert(VecSize % SubVecSize == 0, "VecSize not iterable by ItVecSize");

        __device__ Iterator(VecT<DataT, VecSize>& parent, uint32_t startIndex = 0)
            : mIndex(startIndex)
            , mParent(parent)
        {
        }

        __device__ Iterator(VecT<DataT, VecSize> const& parent, uint32_t startIndex = 0)
            : mIndex(startIndex)
            , mParent(const_cast<VecT<DataT, VecSize>&>(parent))
        {
        }

        __device__ constexpr static inline int32_t range()
        {
            return Traits::Range;
        }

        __device__ inline int32_t index() const
        {
            return mIndex;
        }

        __device__ inline typename Traits::ItVecT const& operator*() const
        {
            return *reinterpret_cast<typename Traits::ItVecT const*>(
                &(mParent[mIndex * SubVecSize]));
        }

        __device__ inline typename Traits::ItVecT& operator*()
        {
            return *reinterpret_cast<typename Traits::ItVecT*>(&(mParent[mIndex * SubVecSize]));
        }

        __device__ inline Iterator<SubVecSize>& operator++(int)
        {
            mIndex++;
            return *this;
        }

        __device__ inline Iterator<SubVecSize>& operator++()
        {
            mIndex++;
            return *this;
        }

        __device__ inline Iterator<SubVecSize>& operator+=(int i)
        {
            mIndex += i;
            return *this;
        }

        __device__ inline Iterator<SubVecSize>& operator--()
        {
            mIndex--;
            return *this;
        }

        __device__ inline Iterator<SubVecSize>& operator--(int)
        {
            mIndex--;
            return *this;
        }

        __device__ inline Iterator<SubVecSize>& operator-=(int i)
        {
            mIndex -= i;
            return *this;
        }

        __device__ inline Iterator<SubVecSize> next() const
        {
            return Iterator<SubVecSize>(mParent, mIndex + 1);
        }

        __device__ inline Iterator<SubVecSize> prev() const
        {
            return Iterator<SubVecSize>(mParent, mIndex - 1);
        }

        __device__ bool valid() const
        {
            return (mIndex >= 0) && (mIndex < Traits::Range);
        }
    };

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> begin()
    {
        return Iterator<SubVecSize>(*this);
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> begin() const
    {
        return Iterator<SubVecSize>(*this);
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> end()
    {
        return Iterator<SubVecSize>(*this, Iterator<SubVecSize>::range());
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> end() const
    {
        return Iterator<SubVecSize>(*this, Iterator<SubVecSize>::range());
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> iterator(uint32_t startIndex = 0)
    {
        return Iterator<SubVecSize>(*this, startIndex);
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> iterator(uint32_t startIndex = 0) const
    {
        return Iterator<SubVecSize>(*this, startIndex);
    }
};

// V registers
using VRegI8x1  = VecT<int8_t, 1>; // Single i8 register
using VRegI8x2  = VecT<int8_t, 2>; // Two i8 registers
using VRegI8x4  = VecT<int8_t, 4>; // ...
using VRegI8x8  = VecT<int8_t, 8>; //
using VRegI8x16 = VecT<int8_t, 16>; //
using VRegI8x32 = VecT<int8_t, 32>; // 32 i8 registers

using VRegU8x1  = VecT<uint8_t, 1>; // Single u8 register
using VRegU8x2  = VecT<uint8_t, 2>; // Two ui8 registers
using VRegU8x4  = VecT<uint8_t, 4>; // ...
using VRegU8x8  = VecT<uint8_t, 8>; //
using VRegU8x16 = VecT<uint8_t, 16>; //
using VRegU8x32 = VecT<uint8_t, 32>; // 32 u8 registers

using VRegI32x1  = VecT<int32_t, 1>; // Single i32 register
using VRegI32x2  = VecT<int32_t, 2>; // Two i32 registers
using VRegI32x4  = VecT<int32_t, 4>; // ...
using VRegI32x8  = VecT<int32_t, 8>; //
using VRegI32x16 = VecT<int32_t, 16>; //
using VRegI32x32 = VecT<int32_t, 32>; // 32 i32 registers

using VRegU32x1  = VecT<uint32_t, 1>; // Single u32 register
using VRegU32x2  = VecT<uint32_t, 2>; // Two u32 registers
using VRegU32x4  = VecT<uint32_t, 4>; // ...
using VRegU32x8  = VecT<uint32_t, 8>; //
using VRegU32x16 = VecT<uint32_t, 16>; //
using VRegU32x32 = VecT<uint32_t, 32>; // 32 u32 registers

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

// Note: In assembly, fp64 registers are actually 2 x fp32 regs
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

using AccRegU32x1  = VecT<uint32_t, 1>;
using AccRegU32x2  = VecT<uint32_t, 2>;
using AccRegU32x4  = VecT<uint32_t, 4>;
using AccRegU32x8  = VecT<uint32_t, 8>;
using AccRegU32x16 = VecT<uint32_t, 16>;
using AccRegU32x32 = VecT<uint32_t, 32>;

using AccRegF32x1  = VecT<float32_t, 1>;
using AccRegF32x2  = VecT<float32_t, 2>;
using AccRegF32x4  = VecT<float32_t, 4>;
using AccRegF32x8  = VecT<float32_t, 8>;
using AccRegF32x16 = VecT<float32_t, 16>;
using AccRegF32x32 = VecT<float32_t, 32>;

// Note: In assembly, fp64 registers are actually 2 x fp32 regs
using AccRegF64x1  = VecT<float64_t, 1>;
using AccRegF64x2  = VecT<float64_t, 2>;
using AccRegF64x4  = VecT<float64_t, 4>;
using AccRegF64x8  = VecT<float64_t, 8>;
using AccRegF64x16 = VecT<float64_t, 16>;
using AccRegF64x32 = VecT<float64_t, 32>;

// clang-format off

// Data layout meta-tags
struct row_major{};
struct col_major{};

// Fragment usage meta-tags
struct matrix_a{};
struct matrix_b{};
struct accumulator{};

// Memory meta-tags
struct globalMem{};
struct ldsMem{};

// clang-format on

#endif // WMMA_TYPES_H
