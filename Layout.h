#ifndef WMMA_LAYOUT_H
#define WMMA_LAYOUT_H

#include <hip/hip_runtime.h>

#include "Types.h"

template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
struct amdgcn_io_traits;

/* Layouts

These layouts are based in Matrix Space. They are to map each of the wave lanes into
corresponding row / col coordinates for a particular memory layout.

For example, the A matrix loads columns of size BlockDim in the K direction. The B matrix
loads rows of size BlockDim in the K direction.

Each of these layouts is indexed differently, especially when different datatypes and load
widths are used. These classes are intended to address these matrix space indexing challenges.

*/

namespace Layout
{

    ////////////// Col /////////////////////////
    /*
        Every register holds k columns of size blockDim.

        Elements 0.....31 32.....64
                _______________
        Reg0    |  C0   |   C1  |
        Reg1    |  C2   |   C3  |
        ...       ...      ...
     */
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t ElementsPerThread>
    struct Col;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Col<BlockDim, BlockK, DataT, row_major, ElementsPerThread>
    {
        using Traits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            uint32_t rowOffset = ((threadIdx.x) % BlockDim) * ldm;
            uint32_t colOffset = (threadIdx.x / BlockDim) * ElementsPerThread % Traits::KPerIO;

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return i * Traits::KPerIO; // Shift K
        }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Col<BlockDim, BlockK, DataT, col_major, ElementsPerThread>
    {
        using Traits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            uint32_t rowOffset = ((threadIdx.x * ElementsPerThread) % BlockDim);
            uint32_t colOffset
                = (threadIdx.x * ElementsPerThread / BlockDim) % Traits::KPerIO * ldm;

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return i * Traits::KPerIO * ldm; // Shift K
        }
    };

    ////////////// Row /////////////////////////
    // Every register holds 2 x 32 elements
    // of consecutive K strides. In this case,
    // K strides are matrix rows of size BlockDim
    //
    // Elements 0.....31 32.....64
    //          _______________
    // Reg0    |  R0   |   R1  |
    // Reg1    |  R2   |   R3  |
    // ...        ...      ...

    // Row layout is the transpose of column layout
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t ElementsPerThread>
    struct Row;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Row<BlockDim, BlockK, DataT, row_major, ElementsPerThread>
        : public Col<BlockDim, BlockK, DataT, col_major, ElementsPerThread>
    {
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Row<BlockDim, BlockK, DataT, col_major, ElementsPerThread>
        : public Col<BlockDim, BlockK, DataT, row_major, ElementsPerThread>
    {
    };

    ////////////// Row4T /////////////////////////
    // Every 4 registers holds 8 x 32 elements rows
    //
    // Elements 0.....31 32.....64
    //          _______________
    // Reg0    |  R0   |   R4  |
    // Reg1    |  R1   |   R5  |
    // Reg2    |  R2   |   R6  |
    // Reg3    |  R3   |   R7  |
    //          _______________
    // Reg4    |  R8   |   R12 |
    //            ...      ...
    //
    // Similar to row layout, however register halves are transposed
    // in every group of 4 registers.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t ElementsPerThread>
    struct Row4T;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Row4T<BlockDim, BlockK, DataT, row_major, ElementsPerThread>
    {
        using Traits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
        enum : uint32_t
        {
            RCount = 4
        };

        static_assert(Traits::IOCount % RCount == 0,
                      "Need a minimum of 4 registers for this layout");

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            // Initialize starting offsets.
            uint32_t rowOffset = (threadIdx.x / BlockDim) % Traits::KPerIO * RCount * ldm;
            uint32_t colOffset = (threadIdx.x * ElementsPerThread % BlockDim);

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return (((i / RCount) * RCount * Traits::KPerIO + (i % RCount))) * ldm;
        }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t ElementsPerThread>
    struct Row4T<BlockDim, BlockK, DataT, col_major, ElementsPerThread>
    {
        using Traits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
        enum : uint32_t
        {
            RCount = 4
        };

        static_assert(Traits::UnpackedRegisterCount % RCount == 0,
                      "Need a minimum of 4 registers for this layout");

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            // Initialize starting offsets.
            uint32_t rowOffset
                = (threadIdx.x * ElementsPerThread / BlockDim) % Traits::KPerIO * RCount;
            uint32_t colOffset = (threadIdx.x % BlockDim) * ldm; // K Id

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return (((i / RCount) * RCount * Traits::KPerIO + i % RCount));
        }
    };

} // namespace Layout

#endif // WMMA_LAYOUT_H
