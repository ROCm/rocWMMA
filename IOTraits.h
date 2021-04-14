#ifndef WMMA_IO_TRAITS_H
#define WMMA_IO_TRAITS_H

#include "Constants.h"
#include "Utils.h"

/*
* The following class is intended to define the packing traits
* for particular datatypes. We consider that WMMA uses packed
* registers. The pack ratio is how many registers resulting from
* raw IO are packed together while used in WMMA.
*/

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
struct PackTraits<__half>
{
    enum : uint32_t
    {
        PackRatio = 2 // 2 Elements combine to one
    };

    using UnpackedT = __half;
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

/*
* The following class is intended to provide insightful suggestions for
* IO vector widths. Given a certain size of block, there are finite
* elements to load which will affect appropriate vector widths. We also
* consider that WMMA uses packed registers. For example in fp16 if the
* IO count is only 1, the packed reg count is 0.5, which is not useful
* for WMMA. We need to expect 1 full packed register at a minimum.
*
* We start testing at a default width of 8. If we cannot satisfy the
* minimum requirements, we recurse into a reduced width of 4 and so on
* until the min reqs are satisfied.
*/

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t TestWidth = 8>
struct VecWidthTraits
{
    enum : uint32_t
    {
        IOCount    = (BlockDim * BlockK) / (AMDGCN_WAVE_SIZE * TestWidth),
        MinIOCount = PackTraits<DataT>::PackRatio,
        MaxElementsPerThread
        = IOCount >= MinIOCount
              ? TestWidth
              : VecWidthTraits<BlockDim, BlockK, DataT, TestWidth / 2>::MaxElementsPerThread
    };
};

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct VecWidthTraits<BlockDim, BlockK, DataT, 0>
{
    enum : uint32_t
    {
        IOCount              = 0,
        MinIOCount           = 1,
        MaxElementsPerThread = 0
    };
};

/*
* The following class provides IO meta-data that is used
* to provide static information used in inference and controlling
* certain aspects of WMMA. Static asserts are also performed in this
* class to indicate specific logical issues when implementing IO
* functionality.
*/
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
