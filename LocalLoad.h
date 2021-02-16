#ifndef WMMA_LOCAL_LOAD_H
#define WMMA_LOCAL_LOAD_H

#include "Types.h"
#include "Layout.h"

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_local_load;

template <>
struct amdgcn_local_load<float32_t, 1>
{
    using LoadT = VRegF32x1;
    __device__ static inline auto exec(float const* localPtr, index_t offset) -> LoadT
    {
        return LoadT(localPtr[offset]);
    }
};

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_local_load_dword_traits;

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_local_load_dword_traits<BlockDim, BlockK, float32_t>
{
    using DataT  = float32_t; // Float data
    using Loader = amdgcn_local_load<DataT, 1>; // Load DWORD, one float per thread
    using LoadT  = typename Loader::LoadT; // Output register type per load

    enum : uint32_t
    {
        StridesPerLoad
        = AMDGCN_WAVE_SIZE / BlockDim, // Number of consecutive strides of BlockDim per load
        LoadCount = ceilDiv(BlockDim * BlockK,
                            AMDGCN_WAVE_SIZE) // Number of loads required for BlockDim * BlockK
    };

    static_assert(LoadCount >= 1, "Loads must fill at least one register");

    using ResultT = VecT<DataT, LoadCount>; // Collection of registers for total load
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout = row_major>
struct amdgcn_local_load_dword_DxK
{
    // LDS layout will be treated as rows, in row-major format for best performance
    struct Traits : public amdgcn_local_load_dword_traits<BlockDim, BlockK, DataT>
    {
        using LayoutT = typename Layout::template Row<BlockDim, BlockK, DataT, DataLayout>;
    };

    // Extract traits
    using Loader  = typename Traits::Loader;
    using LoadT   = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;
    using LayoutT = typename Traits::LayoutT;

    __device__ static auto exec(DataT const* localPtr, uint32_t ldm) -> ResultT
    {
        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        ResultT result;
#pragma unroll
        for(uint32_t i = 0; i < Traits::LoadCount; ++i)
        {
            LoadT loadResult = Loader::exec(localPtr,
                                            (initOffset + LayoutT::iterativeOffset(i, ldm))
                                                );
            result[i]        = *(loadResult);
        }
        return result;
    }
};

#endif // WMMA_LOCAL_LOAD_H