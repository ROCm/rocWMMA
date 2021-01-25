#ifndef WMMA_H_
#define WMMA_H_

#include "BufferLoad.h"
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
        enum : uint32_t
        {
            // Matrix A loads by col size BlockM
            // Matrix B / C load by row size BlockN
            LeadingDim = std::is_same<MatrixT, matrix_a>::value ? BlockM : BlockN,

            // Matrix C loads BlockM rows size BlockN
            // Matrix A and B load BlockK strides of leading dim.
            KDim            = std::is_same<MatrixT, accumulator>::value ? BlockM : BlockK,
            StorageElements = BufferLoad::Traits::LoadCount
        };

        // TODO: maybe add some intelligence here on choice of load (dword, dwordx2, dwordx4).
        // Could differentiate on MatrixT or any dimension if needed.
        using BufferLoad = amdgcn_buffer_load_dword_DxK<MatrixT, LeadingDim, KDim, DataT, LayoutT>;
        //using BufferStore = amdgcn_buffer_store_dword_DxK<MatrixT, LEADING_DIM, K_DIM, DataT, LayoutT>;
        using LoadT    = typename BufferLoad::Traits::LoadT;
        using StorageT = typename BufferLoad::Traits::ResultT;

        __device__ inline LoadT& operator[](uint32_t index)
        {
            return mStorage[index];
        }

        __device__ inline StorageT& operator*()
        {
            return mStorage;
        }

        StorageT mStorage;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>& a,
                                     const DataT*                                               p,
                                     unsigned                                                   ldm)
    {
        using Loader = typename decltype(a)::BufferLoad;
        *a           = BufferLoad::exec(p, ldm);
    }

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename LayoutT>
    void mma_sync(fragment<matrix_a, BlockM, BlockN, BlockK, DataT>& d
                      fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutT>& a,
                  fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutT>&     b,
                  fragment<matrix_a, BlockM, BlockN, BlockK, DataT>&              c)
    {
    }

} // namespace wmma

#endif // WMMA_H_
