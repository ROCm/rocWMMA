#ifndef WMMA_IMPL_H_
#define WMMA_IMPL_H_

#include <type_traits>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "CoopLoad.h"
#include "IOBroadcast.h"
#include "MFMA.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "WMMA.h"

#include "Types.h"

namespace wmma
{
    // fragment implementations
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline DataT&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator[](uint32_t index)
    {
        return mStorage[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline auto fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator*() ->
        typename Traits::StorageT&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline DataT const&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator[](uint32_t index) const
    {
        return mStorage[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline auto
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator*() const ->
        typename Traits::StorageT const&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::leadingDim()
    {
        return Traits::LeadingDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::kDim()
    {
        return Traits::KDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::elementCount()
    {
        return Traits::ElementCount;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::registerCount()
    {
        return Traits::RegisterCount;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                      DataT                                                         value)
    {
        using FragT       = typename std::decay<decltype(frag)>::type;
        using Broadcaster = PackedBroadcastRegs<DataT, FragT::registerCount()>;

        (*frag) = Broadcaster::exec(value);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              typename MemT>
    __device__ void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                                  data,
                         uint32_t                                                      ldm)
    {
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = IOConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Loader = typename Config::GlobalLoader;
        using Packer = typename Config::Packer;

        // Pack and store into frag
        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");
        (*frag) = Packer::exec(Loader::exec(data, ldm));
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename MemT>
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                     const DataT*                                      data,
                                     uint32_t                                          ldm,
                                     layout_t                                          layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        if(layout == layout_t::mem_row_major)
        {
            load_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, row_major, MemT>(
                reinterpret_cast<FragRowMajor&>(frag), data, ldm);
        }
        else
        {
            load_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, col_major, MemT>(
                reinterpret_cast<FragColMajor&>(frag), data, ldm);
        }
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm)
    {
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT     = typename std::decay<decltype(frag)>::type;
        using Loader    = amdgcn_cooperative_load_dword_DxK<MatrixT,
                                                         FragT::leadingDim(),
                                                         FragT::kDim(),
                                                         DataT,
                                                         DataLayout>;
        using WaveSpace = _MappingUtil::WaveSpace;

        // Pack and store into frag
        using Packer = Pack<
            DataT,
            amdgcn_io_traits<FragT::leadingDim(), FragT::kDim(), DataT>::UnpackedRegisterCount>;
        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        // Cooperative load will split the global load amongst all waves in the workgroup
        // because they will all be using the same tile.
        HIP_DYNAMIC_SHARED(DataT, localMemPtr);
        auto waveCount = WaveSpace::workgroupDim();
        (*frag)        = Packer::exec(Loader::exec(data,
                                            ldm,
                                            localMemPtr,
                                            FragT::leadingDim(),
                                            std::get<0>(waveCount),
                                            std::get<1>(waveCount)));
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              typename MemT>
    __device__ void
        store_matrix_sync(DataT*                                                              data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                            ldm)
    {
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT    = typename std::decay<decltype(frag)>::type;
        using Config   = IOConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Storer   = typename Config::GlobalStorer;
        using Unpacker = typename Config::Unpacker;

        // Unpack and scatter
        static_assert(std::is_same<typename FragT::Traits::StorageT,
                                   typename Unpacker::Traits::InputT>::value,
                      "Fragment storage type and packed types do not match");
        Storer::exec(data, Unpacker::exec(*frag), ldm);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename MemT>
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        if(layout == layout_t::mem_row_major)
        {
            store_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, row_major, MemT>(
                data, reinterpret_cast<FragRowMajor const&>(frag), ldm);
        }
        else
        {
            store_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, col_major, MemT>(
                data, reinterpret_cast<FragColMajor const&>(frag), ldm);
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

#endif // WMMA_IMPL_H_
