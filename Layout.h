#ifndef WMMA_LAYOUT_H
#define WMMA_LAYOUT_H

#include <hip/hip_runtime.h>

#include "Types.h"

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_buffer_load_dword_traits;

namespace Layout
{

    ////////////// Col /////////////////////////
    // Every register holds 2 x 32 elements
    // of consecutive K strides. In this case,
    // K strides are matrix columns of size BlockDim
    //
    // Elements 0.....31 32.....64
    //          _______________
    // Reg0    |  C0   |   C1  |
    // Reg1    |  C2   |   C3  |
    //  ...       ...      ...
    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
    struct Col;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Col<BlockDim, BlockK, DataT, row_major>
    {
        using Traits = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            uint32_t rowOffset = (threadIdx.x % BlockDim) * ldm;
            uint32_t colOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad;

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return i * Traits::StridesPerLoad; // Shift K
        }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Col<BlockDim, BlockK, DataT, col_major>
    {
        using Traits = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            uint32_t rowOffset = (threadIdx.x % BlockDim);
            uint32_t colOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad * ldm;

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return i * Traits::StridesPerLoad * ldm; // Shift K
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
    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
    struct Row;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Row<BlockDim, BlockK, DataT, row_major> : public Col<BlockDim, BlockK, DataT, col_major>
    {
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Row<BlockDim, BlockK, DataT, col_major> : public Col<BlockDim, BlockK, DataT, row_major>
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
    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
    struct Row4T;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Row4T<BlockDim, BlockK, DataT, row_major>
    {
        using Traits = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;
        enum : uint32_t
        {
            RCount = 4
        };

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            // Initialize starting offsets.
            uint32_t rowOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad * RCount * ldm;
            uint32_t colOffset = (threadIdx.x % BlockDim);

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return (((i / RCount) * RCount * Traits::StridesPerLoad + (i % RCount))) * ldm;
        }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    struct Row4T<BlockDim, BlockK, DataT, col_major>
    {
        using Traits = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;
        enum : uint32_t
        {
            RCount = 4
        };

        __device__ static inline uint32_t initialOffset(uint32_t ldm)
        {
            // Initialize starting offsets.
            uint32_t rowOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad * RCount;
            uint32_t colOffset = (threadIdx.x % BlockDim) * ldm; // K Id

            return rowOffset + colOffset;
        }

        __device__ static inline uint32_t iterativeOffset(uint32_t i, uint32_t ldm)
        {
            return (((i / RCount) * RCount * Traits::StridesPerLoad + i % RCount));
        }
    };

    // Selector for K layout based on MatrixT
    template <typename MatrixT>
    struct KLayout;

    // Matrix A loads matrix columns of size BlockDim in the K direction
    template <>
    struct KLayout<matrix_a>
    {
        template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
        using LayoutT = Layout::Col<BlockDim, BlockK, DataT, DataLayout>;
    };

    // Matrix B loads matrix rows of size BlockDim in the K direction
    template <>
    struct KLayout<matrix_b>
    {
        template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
        using LayoutT = Layout::Row<BlockDim, BlockK, DataT, DataLayout>;
    };

    // Accumulator loads matrix rows of size BlockDim in the K direction,
    // with the rows transposed each group of 4 registers.
    template <>
    struct KLayout<accumulator>
    {
        template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
        using LayoutT = Layout::Row4T<BlockDim, BlockK, DataT, DataLayout>;
    };

} // namespace Layout

#endif // WMMA_LAYOUT_H
