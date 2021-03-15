#ifndef WMMA_TYPES_H
#define WMMA_TYPES_H

#include <hip/hip_runtime.h>
#include <type_traits>

// General types
using float32_t = float;
using float16_t = _Float16;
using float64_t = double;
using int32_t   = int;
using index_t   = int32_t;

namespace std
{
    template <>
    struct is_floating_point<float16_t> : public true_type
    {
    };

    inline ostream& operator<<(ostream& stream, const float16_t val)
    {
        return stream << static_cast<float>(val);
    }
} // namespace std

// Vector types
template <typename T, int Elements>
using _VecT =
    typename std::conditional<Elements == 1, T, T __attribute__((ext_vector_type(Elements)))>::type;

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

// Packed type conversion
template <typename DataT>
using PackedType =
    typename std::conditional<std::is_floating_point<DataT>::value, float32_t, int32_t>::type;

// Vector wrapper for element access.
template <typename T, uint32_t VecSize>
struct VecT;

template <typename T, uint32_t VecSize>
struct __align__(4) VecT
{
    using VecType = _VecT<T, VecSize>;
    using Type    = T;

    union DataT
    {
        VecType v; // Vector representation
        T       e[VecSize]; // Element representation
    };

    __device__ VecT()
    {
        mData.v = VecType(0);
    }

    __device__ VecT(VecT const& other)
    {
        mData.v = other.mData.v;
    }

    __device__ VecT(VecType const& other)
    {
        mData.v = other;
    }

    __device__ VecT(VecType && other)
    {
        mData.v = std::move(other);
    }

    __device__ T& operator[](uint32_t index)
    {
        return mData.e[index];
    }

    __device__ VecType& operator*()
    {
        return mData.v;
    }

    __device__ VecType& v()
    {
        return *(*this);
    }

    __device__ T& e(uint32_t index)
    {
        return (*this)[index];
    }

    __device__ T const& operator[](uint32_t index) const
    {
        return mData.e[index];
    }

    __device__ VecType const& operator*() const
    {
        return mData.v;
    }

    __device__ VecType const& v() const
    {
        return *(*this);
    }

    __device__ T const& e(uint32_t index) const
    {
        return (*this)[index];
    }

    __device__ constexpr static inline uint32_t size()
    {
        return VecSize;
    }

    template <uint32_t SubVecSize>
    struct Iterator
    {
        using ItVecT = _VecT<T, SubVecSize>;
        enum : uint32_t
        {
            Range = VecSize / SubVecSize
        };

        static_assert(VecSize % SubVecSize == 0, "VecSize not iterable by ItVecSize");

        __device__ Iterator(VecT<T, VecSize>& parent, uint32_t startIndex = 0)
            : mIndex(startIndex)
            , mParent(parent)
        {
        }

        __device__ Iterator(VecT<T, VecSize> const& parent, uint32_t startIndex = 0)
            : mIndex(startIndex)
            , mParent(const_cast<VecT<T, VecSize>&>(parent))
        {
        }

        int32_t           mIndex = 0;
        VecT<T, VecSize>& mParent;

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

private:
    DataT mData;
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
