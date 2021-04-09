#ifndef WMMA_TYPES_H
#define WMMA_TYPES_H

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <type_traits>

// Native types
using float32_t = float;
using float16_t = _Float16;
using float64_t = double;
using int32_t   = int;
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
    using StorageT = _VecT<T, VecSize>;
    using DataT    = T;

    union
    {
        static_assert(sizeof(StorageT) == sizeof(DataT[VecSize]),
                      "Unable to vectorize with StorageT");
        StorageT v; // Vector representation
        DataT    e[VecSize]; // Element array representation
    };

    __device__ VecT()  = default;
    __device__ ~VecT() = default;

    __device__ VecT(VecT const& other)
    {
        v = other.v;
    }

    __device__ VecT(StorageT const& other)
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
        using ItVecT = _VecT<DataT, SubVecSize>;
        enum : uint32_t
        {
            Range = VecSize / SubVecSize
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

        int32_t               mIndex = 0;
        VecT<DataT, VecSize>& mParent;

        __device__ inline ItVecT const& operator*() const
        {
            return *reinterpret_cast<ItVecT const*>(&(mParent[mIndex * SubVecSize]));
        }

        __device__ inline ItVecT& operator*()
        {
            return *reinterpret_cast<ItVecT*>(&(mParent[mIndex * SubVecSize]));
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
            return (mIndex >= 0) && (mIndex < Range);
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
        return Iterator<SubVecSize>(*this, Iterator<SubVecSize>::Range);
    }

    template <uint32_t SubVecSize = 1>
    __device__ inline Iterator<SubVecSize> end() const
    {
        return Iterator<SubVecSize>(*this, Iterator<SubVecSize>::Range);
    }
};

// V registers
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

// Acc registers
using AccRegF32x1  = VecT<float32_t, 1>;
using AccRegF32x2  = VecT<float32_t, 2>;
using AccRegF32x4  = VecT<float32_t, 4>;
using AccRegF32x8  = VecT<float32_t, 8>;
using AccRegF32x16 = VecT<float32_t, 16>;
using AccRegF32x32 = VecT<float32_t, 32>;

// Meta-tags
// Matrices
struct row_major;
struct col_major;
struct matrix_a;
struct matrix_b;
struct accumulator;
struct common;

// Memory
struct globalMem;
struct ldsMem;

#endif // WMMA_TYPES_H
