#ifndef WMMA_LOCAL_STORE_H
#define WMMA_LOCAL_STORE_H

#include "Types.h"
#include "Layout.h"

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_local_store;

template <>
struct amdgcn_local_store<float32_t, 1>
{
    using StoreT = VRegF32x1;
    __device__ static inline void exec(float* localPtr, StoreT const& data, index_t offset)
    {
        localPtr[offset] = *data;
    }
};

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_local_store_dword_traits;

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_local_store_dword_traits<BlockDim, BlockK, float32_t>
{
    using DataT  = float32_t; // Float data
    using Storer = amdgcn_local_store<DataT, 1>; // Load DWORD, one float per thread
    using StoreT  = typename Storer::StoreT; // Output register type per load

    enum : uint32_t
    {
        StridesPerStore
        = AMDGCN_WAVE_SIZE / BlockDim / 1,   // Number of consecutive strides of BlockDim per load
        StoreCount = ceilDiv(BlockDim * BlockK,
                            AMDGCN_WAVE_SIZE) // Number of loads required for BlockDim * BlockK
    };

    static_assert(StoreCount >= 1, "Loads must fill at least one register");

    using ResultT = VecT<DataT, StoreCount>; // Collection of registers for total load
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout = row_major>
struct amdgcn_local_store_dword_DxK
{
    // LDS layout will be treated as rows, in row-major format for best performance
    struct Traits : public amdgcn_local_store_dword_traits<BlockDim, BlockK, DataT>
    {
        using LayoutT = typename Layout::template Row<BlockDim, BlockK, DataT, DataLayout>;
    };

    // Extract traits
    using Storer  = typename Traits::Storer;
    using StoreT   = typename Traits::StoreT;
    using ResultT = typename Traits::ResultT;
    using LayoutT = typename Traits::LayoutT;

    __device__ static void exec(DataT* localPtr, ResultT const& data, uint32_t ldm)
    {
        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
#pragma unroll
        for(uint32_t i = 0; i < Traits::StoreCount; ++i)
        {
            Storer::exec(localPtr, data[i], (initOffset + LayoutT::iterativeOffset(i, ldm)));
        }
    }
};

#endif // WMMA_LOCAL_STORE_H