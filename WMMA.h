#ifndef WMMA_H_
#define WMMA_H_

#include <type_traits>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "MFMA.h"

#include "Types.h"

namespace wmma
{

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT = void>
    struct fragment
    {
        struct Traits
        {
            enum : uint32_t
            {
                // Matrix A loads by col size BlockM
                // Matrix B / C load by row size BlockN
                LeadingDim = std::is_same<MatrixT, matrix_a>::value ? BlockM : BlockN,

                // Matrix C loads BlockM rows size BlockN
                // Matrix A and B load BlockK strides of leading dim.
                KDim = std::is_same<MatrixT, accumulator>::value ? BlockM : BlockK,
            };

            // TODO: maybe add some intelligence here on choice of load (dword, dwordx2, dwordx4).
            // Could differentiate on MatrixT or any dimension if needed.
            using BufferLoad
                = amdgcn_buffer_load_dword_DxK<MatrixT, LeadingDim, KDim, DataT, LayoutT>;

            using BufferStore
                = amdgcn_buffer_store_dword_DxK<MatrixT, LeadingDim, KDim, DataT, LayoutT>;

            static_assert(std::is_same<typename BufferLoad::Traits::ResultT,
                                       typename BufferStore::Traits::ResultT>::value,
                          "Mismatched BufferLoad and BufferStore types");

            using StorageT    = typename BufferLoad::Traits::ResultT;
            using DataLayoutT = LayoutT;

            enum : uint32_t
            {
                StorageElements = StorageT::size(),
            };
        };

        __device__ inline DataT& operator[](uint32_t index)
        {
            return mStorage[index];
        }

        __device__ inline typename Traits::StorageT& operator*()
        {
            return mStorage;
        }

        __device__ inline DataT const& operator[](uint32_t index) const
        {
            return mStorage[index];
        }

        __device__ inline typename Traits::StorageT const& operator*() const
        {
            return mStorage;
        }

        __device__ constexpr static inline uint32_t num_elements()
        {
            return Traits::StorageElements;
        }

        typename Traits::StorageT mStorage;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ void fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>& frag,
                                  DataT                                                      value)
    {
#pragma unroll
        for(uint32_t i = 0; i < frag.num_elements(); i++)
        {
            frag[i] = value;
        }
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>& frag,
                         const DataT*                                               data,
                         uint32_t                                                   ldm)
    {
        static_assert(!std::is_same<LayoutT, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT      = typename std::decay<decltype(frag)>::type;
        using BufferLoad = typename FragT::Traits::BufferLoad;
        (*frag)          = BufferLoad::exec(data, ldm);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                     const DataT*                                      data,
                                     uint32_t                                          ldm,
                                     layout_t                                          layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        if(layout == layout_t::mem_row_major)
        {
            load_matrix_sync(reinterpret_cast<FragRowMajor&>(frag), data, ldm);
        }
        else
        {
            load_matrix_sync(reinterpret_cast<FragColMajor&>(frag), data, ldm);
        }
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ void
        store_matrix_sync(DataT*                                                           data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT> const& frag,
                          uint32_t                                                         ldm)
    {
        static_assert(!std::is_same<LayoutT, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT       = typename std::decay<decltype(frag)>::type;
        using BufferStore = typename FragT::Traits::BufferStore;
        BufferStore::exec((*frag), data, ldm);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        if(layout == layout_t::mem_row_major)
        {
            store_matrix_sync(data, reinterpret_cast<FragRowMajor const&>(frag), ldm);
        }
        else
        {
            store_matrix_sync(data, reinterpret_cast<FragColMajor const&>(frag), ldm);
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB>
    __device__ void mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>&           d,
                             fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const& a,
                             fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const& b,
                             fragment<accumulator, BlockM, BlockN, BlockK, ComputeT> const&     c)
    {
        using MFMA = amdgcn_mfma_MxNxK<InputT, ComputeT, BlockM, BlockN, BlockK>;
        (*d)       = MFMA::exec((*a), (*b), (*c));
    }

} // namespace wmma

#endif // WMMA_H_
