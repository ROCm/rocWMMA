#ifndef WMMA_BUFFER_DESCRIPTOR_H
#define WMMA_BUFFER_DESCRIPTOR_H

#include <hip/hip_runtime.h>

#include "Types.h"

// Meta data for Buffer Descriptor
template<typename T>
struct BufferTraits;

template<>
struct BufferTraits<float32_t>
{
    static constexpr uint32_t DataFormat = 0x4; // 4 = 32 bit elements
    static constexpr uint32_t NumFormat = 0x7;  // 7 = float
};

template<>
struct BufferTraits<uint32_t>
{
    static constexpr uint32_t DataFormat = 0x4; // 4 = 32 bit elements
    static constexpr uint32_t NumFormat = 0x4;  // 4 = uint
};

template<>
struct BufferTraits<int32_t>
{
    static constexpr uint32_t DataFormat = 0x4; // 4 = 32 bit elements
    static constexpr uint32_t NumFormat = 0x5;  // 4 = sint
};

template<typename T>
struct __align__(4) BufferDescriptor
{
    using VecType = v4_i32_t;
    union BufferSrd
    {
        VecType data;
        uint64_t d64[2];
        uint32_t d32[4];
        uint8_t  d128[128];
    };

    __device__ BufferDescriptor(
        T const* addr,                      // Start address of data
        uint32_t stride = 0,                // Data stride (elements)
        bool tidEnable = false,             // Add threadID to stride offset
        uint32_t numRecords = -1,           // Number of bytes, or stride count if used
        bool cacheSwizzle = false,          // Swizzle the cache
        bool swizzle = false,               // Swizzle the data
        uint32_t swizzleIndexStride = 0)    // Swizzle index stride
    {
        mSrd.data = 0;
        mSrd.d64[0] |= 
            ((reinterpret_cast<uint64_t>(addr)) & (uint64_t)0xFFFFFFFFFFFF) |   // Bits[47:0] base address
            (((uint64_t)0x3FFF & (uint64_t)(stride * sizeof(T))) << 48) |       // Bits[61:48] stride (bytes)
            (((uint64_t)0x1 & (uint64_t)(cacheSwizzle)) << 62) |                // Bit[62] cache swizzle enable
            (((uint64_t)0x1 & (uint64_t)(swizzle)) << 63);                      // Bit[63] swizzle enable

        mSrd.d32[2] |= (uint32_t)numRecords;                                    // Bits[95:64]

        mSrd.d32[3] |=
            ((uint32_t)0xFFF & (uint32_t)(0)) |                                 // Bits[107:96] Dest_sel_x,y,z,w = 0
            (((uint32_t)0x7 & (uint32_t)(BufferTraits<T>::NumFormat)) << 12) |  // Bits[110:108] Num format
            (((uint32_t)0xF & (uint32_t)(tidEnable ? (stride >> 14) : BufferTraits<T>::DataFormat)) << 15) | // Bits [114:111] Data format
            (((uint32_t)0x3 & (uint32_t)(swizzleIndexStride)) << 21) |          // Bits[118:117] Index stride
            (((uint32_t)0x1 & (uint32_t)(tidEnable)) << 23);                    // Bits[119] Tid Enable
    }

    __device__ VecType& operator*()
    {
        return mSrd.data;
    }

    __device__ VecType& v()
    {
        return this->operator*();
    }

private:
    BufferSrd mSrd;
};

#endif // WMMA_BUFFER_DESCRIPTOR_H
