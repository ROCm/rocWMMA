#ifndef WMMA_LOCAL_LOAD_H
#define WMMA_LOCAL_LOAD_H

#include "Types.h"
#include "Layout.h"

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_local_load;

template <>
struct amdgcn_local_load<float16_t, 1>
{
    using LoadT = VRegF16x1;
    __device__ static inline auto exec(float16_t const* localPtr, index_t offset) -> LoadT
    {
        return LoadT(localPtr[offset]);
    }
};

template <>
struct amdgcn_local_load<float32_t, 1>
{
    using LoadT = VRegF32x1;
    __device__ static inline auto exec(float32_t const* localPtr, index_t offset) -> LoadT
    {
        return LoadT(localPtr[offset]);
    }
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_local_load_dword_DxK
{
    using Config = LocalConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;

    struct Traits : public TraitsBase
    {
        // Matrix space thread offsets
        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;

        // These traits are per-load
        using Loader = amdgcn_local_load<DataT, Config::ElementsPerThread>;
        using LoadT = typename Loader::LoadT;
        using OutputT = VecT<DataT, TraitsBase::UnpackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* localPtr, uint32_t ldm) -> typename Traits::OutputT
    {
        // Extract traits
        using Loader  = typename Traits::Loader;
        using LoadT   = typename Traits::LoadT;
        using OutputT = typename Traits::OutputT;
        using LayoutT = typename Traits::LayoutT;

        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        OutputT result;
        auto it = result.template begin<LoadT::size()>();

        static_assert(decltype(it)::Range == Traits::IOCount, "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
            *it = *Loader::exec(localPtr, initOffset + LayoutT::iterativeOffset(i, ldm));
            it++;       
        }
        return result;
    }
};

#endif // WMMA_LOCAL_LOAD_H