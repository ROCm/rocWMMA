/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef WMMA_IMPL_H_
#define WMMA_IMPL_H_

#include <type_traits>
#include <utility>

#include "Convert.h"
#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOBarrier.h"
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
    __device__
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::fragment(const fragment& other)
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
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::size()
    {
        return num_elements;
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
        using Config = io_config<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Broadcaster = Broadcast<DataT, Config::IOTraits::UnpackedSize>;
        using Packer      = typename Config::Packer;

        // Sanity checks
        static_assert(std::is_same<typename Broadcaster::Traits::OutputT,
                                   typename Packer::Traits::InputT>::value,
                      "Broadcast output and pack input types do not match");

        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        // Broadcast then pack
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
        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = io_config<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Loader = typename Config::Loader;
        using Packer = typename Config::Packer;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        // Load then pack
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
        using FragT    = typename std::decay<decltype(frag)>::type;
        using Config   = io_config<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Storer   = typename Config::Storer;
        using Unpacker = typename Config::Unpacker;

        // Sanity check
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::StorageT,
                                   typename Unpacker::Traits::InputT>::value,
                      "Fragment storage type and packed types do not match");

        // Unpack and scatter
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

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t splitIndex,
                              uint32_t splitCount)
    {
        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = io_config<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using Packer = typename Config::Packer;
        using CoopLoader = typename Config::CoopLoader;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(
            std::is_same<typename FragT::Traits::StorageT, typename Packer::Traits::OutputT>::value,
            "Fragment storage type and packed types do not match");

        typename CoopLoader::Traits::OutputT unpacked;

        // Each cooperative wave only loads the portion they are responsible for
        // Note: at this point, the output frag is only partially filled with useful data
        CoopLoader::exec(unpacked, data, ldm, splitIndex, splitCount);
        (*frag) = Packer::exec(unpacked);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm,
        uint32_t                                                            splitIndex,
        uint32_t                                                            splitCount)
    {

        using FragT  = typename std::decay<decltype(frag)>::type;
        using Config = io_config<MatrixT, FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;
        using CoopStorer = typename Config::CoopStorer;
        using Unpacker   = typename Config::Unpacker;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::StorageT,
                                   typename Unpacker::Traits::InputT>::value,
                      "Fragment storage type and packed types do not match");

        CoopStorer::exec(data, Unpacker::exec(*frag), ldm, splitIndex, splitCount);
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
        using FragT       = typename std::decay<decltype(frag)>::type;
        using MappingUtil = MappingUtil<FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;

        // Splitting the K direction:
        // Matrix A:
        // - shares work with waves on same row (different col).
        // - waves in different rows work on different blocks
        // Matrix B:
        // - shares work with waves on same col (different row)
        // - waves in different cols work on different blocks
        constexpr auto coopIndex = std::is_base_of<matrix_a, MatrixT>::value ? 1 : 0;
        auto           waveIndex = std::get<coopIndex>(MappingUtil::waveCoord());
        auto           waveCount = std::get<coopIndex>(MappingUtil::workgroupDim());
        load_matrix_coop_sync(frag, data, ldm, waveIndex, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm)
    {

        using FragT       = typename std::decay<decltype(frag)>::type;
        using MappingUtil = MappingUtil<FragT::leadingDim(), FragT::kDim(), DataT, DataLayout>;

        // Splitting the K direction:
        // Matrix A:
        // - shares work with waves on same row (different col).
        // - waves in different rows work on different blocks
        // Matrix B:
        // - shares work with waves on same col (different row)
        // - waves in different cols work on different blocks
        constexpr auto coopIndex = std::is_base_of<matrix_a, MatrixT>::value ? 1 : 0;
        auto           waveIndex = std::get<coopIndex>(MappingUtil::waveCoord());
        auto           waveCount = std::get<coopIndex>(MappingUtil::workgroupDim());

        store_matrix_coop_sync(data, frag, ldm, waveIndex, waveCount);
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

    __device__ void synchronize_workgroup()
    {
        using Barrier = struct amdgcn_barrier;
        Barrier::exec();
    }
} // namespace wmma

#endif // WMMA_IMPL_H_
