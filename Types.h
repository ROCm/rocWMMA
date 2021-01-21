#ifndef WMMA_TYPES_H
#define WMMA_TYPES_H

#include <hip/hip_runtime.h>

// General types
using float32_t = float;
using int32_t = int;
using index_t = int32_t;

// Vector types
template<typename T, int Elements>
using _VecT = typename std::conditional<
    Elements==1,
    T,
    T __attribute__((ext_vector_type(Elements)))>::type;

// Vectors of floats
using v2_f32_t = _VecT<float32_t, 2>;
using v4_f32_t = _VecT<float32_t, 4>;
using v16_f32_t = _VecT<float32_t, 16>;
using v32_f32_t = _VecT<float32_t, 32>;
using v64_f32_t = _VecT<float32_t, 64>;

// Vectors of ints
using v2_i32_t = _VecT<int32_t, 2>;
using v4_i32_t = _VecT<int32_t, 4>;
using v16_i32_t = _VecT<int32_t, 16>;
using v32_i32_t = _VecT<int32_t, 32>;
using v64_i32_t = _VecT<int32_t, 64>;

// Vector wrapper for element access.
template<typename T, int VecSize>
struct VecT;

template<typename T, int VecSize>
struct __align__(4) VecT
{
    using VecType = _VecT<T, VecSize>;
    
    union DataT
    {
        VecType v;     // Vector representation
        T e[VecSize];  // Element representation
    };

    __device__ VecT()
    {
        mData.v = 0;   
    }

    __device__ VecT(VecT const& other)
    {
        mData.v = other.mData.v;   
    }

    __device__ VecT(VecType const& other)
    {
        mData.v = other;   
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
    
private:
    DataT mData;
};

// V registers
using VRegF32x1 = VecT<float32_t, 1>;        // Single f32 register
using VRegF32x2 = VecT<float32_t, 2>;        // Two f32 registers
using VRegF32x4 = VecT<float32_t, 4>;        // ...
using VRegF32x8 = VecT<float32_t, 8>;        // 
using VRegF32x16 = VecT<float32_t, 16>;      // 
using VRegF32x32 = VecT<float32_t, 32>;      // 32 f32 registers

// Acc registers
using AccRegF32x1 = VecT<float32_t, 1>;
using AccRegF32x2 = VecT<float32_t, 2>;
using AccRegF32x4 = VecT<float32_t, 4>;
using AccRegF32x8 = VecT<float32_t, 8>;
using AccRegF32x16 = VecT<float32_t, 16>;
using AccRegF32x32 = VecT<float32_t, 32>;

// Meta-tags
struct row_major;
struct col_major;
struct matrix_a;
struct matrix_b;
struct accumulator;

#endif // WMMA_TYPES_H
