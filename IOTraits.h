#ifndef WMMA_IO_TRAITS_H
#define WMMA_IO_TRAITS_H

#include "Constants.h"
#include "Utils.h"

template <typename DataT>
struct PackTraits;

template <>
struct PackTraits<float16_t>
{
    enum : uint32_t
    {
        PackRatio = 2 // 2 Elements combine to one
    };

    using UnpackedT = float16_t;
    using PackedT   = float32_t;
};

template <>
struct PackTraits<float32_t>
{
    enum : uint32_t
    {
        PackRatio = 1 // No pack
    };

    using UnpackedT = float32_t;
    using PackedT   = float32_t;
};

// IO meta-data
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread = 1>
struct amdgcn_io_traits
{
    enum : uint32_t
    {
        // Number of threads to perform I/O operation
        ThreadsPerIO = AMDGCN_WAVE_SIZE,

        // Number of elements in I/O operation
        ElementsPerIO = ThreadsPerIO * ElementsPerThread,

        // Number of BlockDim strides per I/O operation
        KPerIO = ceilDiv(ElementsPerIO, BlockDim),

        // Number of registers required per I/O operation
        RegistersPerIO = ElementsPerThread,

        // Total number of elements per for the entire block
        ElementCount = BlockDim * BlockK,

        // Total number of I/O operations needed for the entire block
        IOCount = ceilDiv(ElementCount, ElementsPerIO),

        // Total number of registers required for the entire block
        UnpackedRegisterCount = RegistersPerIO * IOCount,

        // Total number of packed registers for the entire block
        PackedRegisterCount = ceilDiv(ElementCount * sizeof(DataT), BYTES_PER_REGISTER),
    };

    static_assert(KPerIO >= 1, "I/O operation must handle at least 1 element in K direction");
    static_assert((ElementsPerIO % BlockDim) == 0,
                  "I/O operation elements not a multiple of BlockDim");
    static_assert((ElementCount % ElementsPerIO) == 0,
                  "I/O element count not divisible into equal operations");
    static_assert((ElementCount * sizeof(DataT) % BYTES_PER_REGISTER) == 0,
                  "Packed elements do not fit equally into registers");
};

#endif // WMMA_IO_TRAITS_H
