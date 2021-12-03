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
#ifndef WMMA_COOP_IMPL_H_
#define WMMA_COOP_IMPL_H_

#include <type_traits>

#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOConfig.h"
#include "IOPack.h"
#include "WMMA.h"

namespace wmma
{
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
                              uint32_t waveIndex,
                              uint32_t waveCount,
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
        CoopLoader::exec(unpacked, data, ldm, waveIndex, waveCount, splitCount);
        (*frag) = Packer::exec(unpacked);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    __device__ inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount)
    {
        loadMatrixCoopSync(frag, data, ldm, waveIndex, waveCount, waveCount);
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
        uint32_t                                                            ldm,
        uint32_t                                                            waveIndex,
        uint32_t                                                            waveCount,
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

        CoopStorer::exec(data, Unpacker::exec(*frag), ldm, waveIndex, waveCount, splitCount);
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
        uint32_t                                                            waveIndex,
        uint32_t                                                            waveCount)
    {
        store_matrix_coop_sync(data, frag, ldm, waveIndex, waveCount, waveCount);
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

        store_matrix_coop_sync(frag, ldm, waveIndex, waveCount);
    }
} // namespace wmma

#endif // WMMA_COOP_IMPL_H_
