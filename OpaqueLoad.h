#ifndef WMMA_OPAQUE_LOAD_H
#define WMMA_OPAQUE_LOAD_H

#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"

template <typename DataT, uint32_t ElementsPerThread>
struct amdgcn_opaque_load
{
    static_assert(ElementsPerThread > 0, "Elements per thread must be greater than 0");

    using LoadT = VecT<typename PackTraits<DataT>::UnpackedT, ElementsPerThread>;
    __device__ static inline auto exec(DataT const* localPtr, index_t offset) -> LoadT
    {
        return LoadT(*reinterpret_cast<typename LoadT::StorageT const*>(&(localPtr[offset])));
    }
};

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class LoadLayout,
          uint32_t ElementsPerThread>
struct amdgcn_opaque_load_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    struct Traits
    {
        // Matrix space thread offsets
        using LayoutT = LoadLayout<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;

        // Raw IO that produce unpacked register data.
        using Loader  = amdgcn_opaque_load<DataT, ElementsPerThread>;
        using LoadT   = typename Loader::LoadT;
        using OutputT = VecT<DataT, IOTraits::UnpackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* localPtr, uint32_t ldm) -> typename Traits::OutputT
    {
        // Extract traits
        using LayoutT = typename Traits::LayoutT;
        using Loader  = typename Traits::Loader;
        using LoadT   = typename Traits::LoadT;
        using OutputT = typename Traits::OutputT;

        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        OutputT result;
        auto    it = result.template begin<LoadT::size()>();

        static_assert(decltype(it)::Range == IOTraits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
        {
            *it = *Loader::exec(localPtr, initOffset + LayoutT::iterativeOffset(i, ldm));
            it++;
        }
        return result;
    }
};

#endif // WMMA_OPAQUE_LOAD_H
