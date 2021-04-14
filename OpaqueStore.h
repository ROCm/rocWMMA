#ifndef WMMA_OPAQUE_STORE_H
#define WMMA_OPAQUE_STORE_H

#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"

template <typename DataT, uint32_t ElementsPerThread>
struct amdgcn_opaque_store
{
    static_assert(ElementsPerThread > 0, "Elements per thread must be greater than 0");

    using StoreT = VecT<typename PackTraits<DataT>::UnpackedT, ElementsPerThread>;
    __device__ static inline void exec(DataT* localPtr, StoreT const& data, index_t offset)
    {
        *reinterpret_cast<typename StoreT::StorageT*>(&(localPtr[offset])) = *data;
    }
};

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t ElementsPerThread>
struct amdgcn_opaque_store_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    struct Traits
    {
        // Matrix space thread offsets
        using LayoutT = StoreLayout<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;

        // These traits are per-io
        using Storer = amdgcn_opaque_store<DataT, ElementsPerThread>;
        using StoreT = typename Storer::StoreT;
        using InputT = VecT<DataT, IOTraits::UnpackedRegisterCount>;
    };

    __device__ static void
        exec(DataT* localPtr, typename Traits::InputT const& incoming, uint32_t ldm)
    {
        // Extract traits
        using Storer  = typename Traits::Storer;
        using StoreT  = typename Traits::StoreT;
        using LayoutT = typename Traits::LayoutT;

        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        auto it = incoming.template begin<StoreT::size()>();
        static_assert(decltype(it)::Range == IOTraits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
        {
            Storer::exec(localPtr, *it, initOffset + LayoutT::iterativeOffset(i, ldm));
            it++;
        }
    }
};

#endif // WMMA_OPAQUE_STORE_H
