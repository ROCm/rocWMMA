#ifndef WMMA_IMPL_H_
#define WMMA_IMPL_H_

#include <type_traits>

#include "Convert.h"
#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOBroadcast.h"
#include "IOConfig.h"
#include "MFMA.h"
#include "MappingUtil.h"
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
    __device__ fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::fragment(const fragment& other)
        : mStorage(other.mStorage)
    {
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator=(
            const fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>& other)
    {
        mStorage = other.mStorage;
        return *this;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline DataT&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator[](uint32_t index)
    {
        return mStorageUnpacked[index];
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
        return mStorageUnpacked[index];
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
        return num_elements;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    __device__ inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::size()
    {
        return Traits::AccessT::Size;
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
        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = OptConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Broadcaster = Broadcast<DataT, Config::IOTraits::UnpackedSize>;
        using Packer      = typename Config::Packer;

        static_assert(std::is_same<typename Broadcaster::Traits::OutputT,
                                   typename Packer::Traits::InputT>::value,
                      "Broadcast output and pack input types do not match");

        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        (*frag) = Packer::exec(Broadcaster::exec(value));
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                                  data,
                         uint32_t                                                      ldm)
    {
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = OptConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Loader = typename Config::GlobalLoader;
        using Packer = typename Config::Packer;

        // Pack and store into frag
        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");
        (*frag) = Packer::exec(Loader::exec(data, ldm));
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
            load_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>(
                reinterpret_cast<FragRowMajor&>(frag), data, ldm);
        }
        else
        {
            load_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>(
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
        store_matrix_sync(DataT*                                                              data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                            ldm)
    {
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        using FragT    = typename std::decay<decltype(frag)>::type;
        using Config   = OptConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Storer   = typename Config::GlobalStorer;
        using Unpacker = typename Config::Unpacker;

        // Unpack and scatter
        static_assert(std::is_same<typename FragT::Traits::StorageT,
                                   typename Unpacker::Traits::InputT>::value,
                      "Fragment storage type and packed types do not match");
        Storer::exec(data, Unpacker::exec(*frag), ldm);
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
            store_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>(
                data, reinterpret_cast<FragRowMajor const&>(frag), ldm);
        }
        else
        {
            store_matrix_sync<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>(
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
        (*d)       = MFMA::exec(*a, *b, *c);
    }

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayout>
    // __device__ void
    //     prefetch_matrix_coop_sync(DataT* ldsData,
    //                             uint32_t ldsLdm,
    //                             fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
    //                             const DataT*                                                  globalData,
    //                             uint32_t                                                      ldm,
    //                             layout_t ldsLayout = DataLayout())
    // {
    //     static_assert(!std::is_same<DataLayout, void>::value,
    //                   "Must provide layout information. Either statically assign data layout in "
    //                   "fragment declaration or use the run-time function overload.");

    //     using FragT       = typename std::decay<decltype(frag)>::type;
    //     using MappingUtil = MappingUtil<FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
    //     using Config   = OptConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
    //     using Packer   = typename Config::Packer;
    //     using Unpacker = typename Config::Unpacker;

    //     using CoopLoader = typename Config::CoopLoader;

    //     // Splitting the K direction:
    //     // Matrix A:
    //     // - shares work with waves on same row (different col).
    //     // - waves in different rows work on different blocks
    //     // Matrix B:
    //     // - shares work with waves on same col (different row)
    //     // - waves in different cols work on different blocks

    //     // coopIndex is used to determine offsets of work that is shared
    //     // auto coopIndex = std::is_same<MatrixT, matrix_a>::value ? [](typename MappingUtil::CoordT const& coord) { return std::get<1>(coord); } :
    //     //                                                           [](typename MappingUtil::CoordT const& coord) { return std::get<0>(coord); };

    //     auto waveCoord    = MappingUtil::waveCoord();
    //     auto waveIndex = std::is_same<MatrixT, matrix_a>::value ? std::get<1>(waveCoord) : std::get<0>(waveCoord);

    //     using WaveSpace = _MappingUtil::WaveSpace;
    //     auto workgroupDim  = WaveSpace::workgroupDim();
    //     auto waveCount = std::is_same<MatrixT, matrix_a>::value ? std::get<1>(waveCoord) : std::get<0>(waveCoord);

    //     typename CoopLoader::Traits::OutputT unpacked = *frag;

    //     // Each cooperative wave only loads the portion they are responsible for
    //     // Note: at this point, the output frag is only partially filled with useful data
    //     CoopLoader::exec(unpacked, data, ldm, waveIndex, waveCount);

    //     (*frag) = Packer::exec(unpacked);
    // }

} // namespace wmma

#endif // WMMA_IMPL_H_
