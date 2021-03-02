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

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_local_store_dword_DxK
{
    using Config = LocalConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;

    struct Traits : public TraitsBase
    {
        // These traits are per-load
        using Storer = amdgcn_local_store<DataT, Config::ElementsPerThread>;
        using StoreT = typename Storer::StoreT;

        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;

        // Input format for entire block.
        // WMMA will load packed results.
        using InputT = VecT<DataT, TraitsBase::UnpackedRegisterCount>;
    };

    __device__ static void exec(DataT* localPtr, typename Traits::InputT const& data, uint32_t ldm)
    {
        // Extract traits
        using Storer  = typename Traits::Storer;
        using StoreT   = typename Traits::StoreT;
        using InputT = typename Traits::InputT;
        using LayoutT = typename Traits::LayoutT;

        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
            Storer::exec(localPtr, data[i], (initOffset + LayoutT::iterativeOffset(i, ldm)));
        }
    }
};

#endif // WMMA_LOCAL_STORE_H