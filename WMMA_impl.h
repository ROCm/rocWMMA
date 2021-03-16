#ifndef WMMA_IMPL_H_
#define WMMA_IMPL_H_

#include <type_traits>

#include "BufferLoad.h"
#include "BufferStore.h"
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
        if(std::is_same<LayoutA, LayoutB>::value)
        {
            HIP_DYNAMIC_SHARED(InputT, localMemPtr);

            using FragAT = typename std::decay<decltype(a)>::type;
            using FragBT = typename std::decay<decltype(b)>::type;

            using ConfigA
                = OptConfig<matrix_a, FragAT::leadingDim(), FragAT::kDim(), InputT, LayoutA>;
            using ConfigB
                = OptConfig<matrix_b, FragBT::leadingDim(), FragBT::kDim(), InputT, LayoutB>;

            typename FragAT::Traits::StorageT AFmt;
            if(std::is_same<LayoutA, row_major>::value)
            {
                using StoreA  = typename ConfigA::CoopStorer;
                using LoadA   = typename ConfigA::LocalLoader;
                using PackA   = typename ConfigA::Packer;
                using UnpackA = typename ConfigA::Unpacker;

                using MappingUtil
                    = MappingUtil<FragAT::leadingDim(), FragAT::kDim(), InputT, LayoutA>;
                auto     waveCoord       = MappingUtil::waveCoord(); // Local to workgroup
                uint32_t competingWaveId = std::get<0>(waveCoord);

                auto ldsOffset = std::make_pair(competingWaveId * FragAT::leadingDim(), 0);
                auto ldsAddr   = MappingUtil::dataCoord(localMemPtr, FragAT::kDim(), ldsOffset);

                StoreA::exec(ldsAddr, UnpackA::exec(*a), FragAT::kDim());
                __syncthreads();

                AFmt = PackA::exec(LoadA::exec(ldsAddr, FragAT::kDim()));
                __syncthreads();
            }
            else
            {
                AFmt = *a;
            }

            typename FragBT::Traits::StorageT BFmt;
            if(std::is_same<LayoutB, col_major>::value)
            {
                using StoreB  = typename ConfigB::CoopStorer;
                using LoadB   = typename ConfigB::LocalLoader;
                using PackB   = typename ConfigB::Packer;
                using UnpackB = typename ConfigB::Unpacker;

                using MappingUtil
                    = MappingUtil<FragBT::leadingDim(), FragBT::kDim(), InputT, LayoutB>;
                auto     waveCoord       = MappingUtil::waveCoord(); // Local to workgroup
                uint32_t competingWaveId = std::get<1>(waveCoord);

                using WaveSpace   = _MappingUtil::WaveSpace;
                auto workgroupDim = WaveSpace::workgroupDim();
                //auto ldsSize = std::make_pair(FragBT::kDim(), std::get<1>(workgroupDim) * FragBT::leadingDim());
                auto ldl = FragBT::kDim();

                auto ldsOffset = std::make_pair(0, competingWaveId * FragBT::leadingDim());
                auto ldsAddr   = MappingUtil::dataCoord(localMemPtr, ldl, ldsOffset);

                StoreB::exec(ldsAddr, UnpackB::exec(*b), ldl);
                __syncthreads();

                BFmt = PackB::exec(LoadB::exec(ldsAddr, ldl));
                __syncthreads();
            }
            else
            {
                BFmt = *b;
            }

            using MFMA = amdgcn_mfma_MxNxK<InputT, ComputeT, BlockM, BlockN, BlockK>;
            (*d)       = MFMA::exec(*AFmt, *BFmt, *c);
        }
        else
        {
            using MFMA = amdgcn_mfma_MxNxK<InputT, ComputeT, BlockM, BlockN, BlockK>;
            (*d)       = MFMA::exec(*a, *b, *c);
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

        using FragT = typename std::decay<decltype(frag)>::type;

        using Config = OptConfig<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;

        using CoopLoader = typename Config::CoopLoader;
        using CoopStorer = typename Config::CoopStorer;

        using Packer   = typename Config::Packer;
        using Unpacker = typename Config::Unpacker;

        using FullLoader = typename Config::LocalLoader;

        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        HIP_DYNAMIC_SHARED(DataT, localMemPtr);
        using MappingUtil  = MappingUtil<FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        auto     waveCoord = MappingUtil::waveCoord(); // Local to workgroup
        uint32_t competingWaveId
            = (std::is_same<MatrixT, matrix_a>::value ? std::get<0>(waveCoord)
                                                      : std::get<1>(waveCoord));

        auto ldsBaseOffset = FragT::leadingDim() * FragT::kDim() * competingWaveId;

        // Cooperative load will split the global load amongst all waves in the workgroup
        // because they will all be using the same tile.
        typename FragT::Traits::StorageT temp;
        CoopLoader::exec(temp, data, ldm);

        CoopStorer::exec(localMemPtr + ldsBaseOffset, temp, FragT::leadingDim());

        __syncthreads();
        *frag = Packer::exec(FullLoader::exec(localMemPtr + ldsBaseOffset, FragT::leadingDim()));

        __syncthreads();
    }

    // template <uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename InputT,
    //           typename ComputeT,
    //           typename LayoutA,
    //           typename LayoutB>
    // __device__ void mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>&           d,
    //                          fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const& a,
    //                          fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const& b,
    //                          fragment<accumulator, BlockM, BlockN, BlockK, ComputeT> const&     c)
    // {
    //     using MFMA = amdgcn_mfma_MxNxK<InputT, ComputeT, BlockM, BlockN, BlockK>;
    //     (*d)       = MFMA::exec((*a), (*b), (*c));
    // }

} // namespace wmma

#endif // WMMA_IMPL_H_
