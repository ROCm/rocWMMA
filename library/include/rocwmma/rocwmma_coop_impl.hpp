/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_COOP_API_IMPL_HPP
#define ROCWMMA_COOP_API_IMPL_HPP

#include "internal/coop_load.hpp"
#include "internal/coop_store.hpp"
#include "rocwmma_coop.hpp"

namespace rocwmma
{
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount,
                              uint32_t splitCount)
    {
        using FragT      = typename std::decay<decltype(frag)>::type;
        using CoopLoader = typename GetIOConfig_t<FragT>::CoopLoader;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::AccessT,
                                   typename CoopLoader::Traits::OutputT>::value,
                      "Fragment access and coop load output types do not match");

        // Load and implicit pack
        // Note: the frag will only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        CoopLoader::exec(frag.mAccess, data, ldm, waveIndex, waveCount, splitCount);
    }

    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex)
    {
        using FragT      = typename std::decay<decltype(frag)>::type;
        using IOShape    = GetIOShape_t<FragT>;
        using CoopLoader = CooperativeLoad<IOShape::BlockDim,
                                           IOShape::KDim,
                                           DataT,
                                           typename IOShape::DataLayout,
                                           typename IOShape::MatrixLayout,
                                           IOShape::VectorWidth>;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::AccessT,
                                   typename CoopLoader::Traits::OutputT>::value,
                      "Fragment access and coop load output types do not match");

        // Load and implicit pack
        // Note: the frag will only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        CoopLoader::template exec<WaveCount, SplitCount>(frag.mAccess, data, ldm, waveIndex);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount)
    {
        load_matrix_coop_sync(frag, data, ldm, waveIndex, waveCount, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm)
    {
        using FragT       = typename std::decay<decltype(frag)>::type;
        using Config      = GetIOConfig_t<FragT>;
        using MappingUtil = typename Config::MappingUtil;

        // Matrix A:
        // - shares work with waves on same row (different col).
        // - waves in different rows work on different blocks
        //
        // Matrix B / Accumulator:
        // - shares work with waves on same col (different row)
        // - waves in different cols work on different blocks
        auto waveIndex = get<Config::CoopIndex>(MappingUtil::waveCoord());
        auto waveCount = get<Config::CoopIndex>(MappingUtil::workgroupDim());
        load_matrix_coop_sync(frag, data, ldm, waveIndex, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm,
        uint32_t                                                            waveIndex,
        uint32_t                                                            waveCount,
        uint32_t                                                            splitCount)
    {

        using FragT      = typename std::decay<decltype(frag)>::type;
        using CoopStorer = typename GetIOConfig_t<FragT>::CoopStorer;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::AccessT,
                                   typename CoopStorer::Traits::InputT>::value,
                      "Fragment access and coop stor input types do not match");

        // Implicit unpack and store
        // Note: the frag is only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        CoopStorer::exec(data, frag.mAccess, ldm, waveIndex, waveCount, splitCount);
    }

    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm,
        uint32_t                                                            waveIndex)
    {

        using FragT      = typename std::decay<decltype(frag)>::type;
        using IOShape    = GetIOShape_t<FragT>;
        using CoopStorer = CooperativeStore<IOShape::BlockDim,
                                            IOShape::KDim,
                                            DataT,
                                            typename IOShape::DataLayout,
                                            typename IOShape::MatrixLayout,
                                            IOShape::VectorWidth>;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(std::is_same<typename FragT::Traits::AccessT,
                                   typename CoopStorer::Traits::InputT>::value,
                      "Fragment access and coop stor input types do not match");

        // Implicit unpack and store
        // Note: the frag is only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        CoopStorer::template exec<WaveCount, SplitCount>(data, frag.mAccess, ldm, waveIndex);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void store_matrix_coop_sync(
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
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm)
    {
        using FragT       = typename std::decay<decltype(frag)>::type;
        using Config      = GetIOConfig_t<FragT>;
        using MappingUtil = typename Config::MappingUtil;

        // Matrix A:
        // - shares work with waves on same row (different col).
        // - waves in different rows work on different blocks
        //
        // Matrix B / Accumulator:
        // - shares work with waves on same col (different row)
        // - waves in different cols work on different blocks
        auto waveIndex = get<Config::CoopIndex>(MappingUtil::waveCoord());
        auto waveCount = get<Config::CoopIndex>(MappingUtil::workgroupDim());
        store_matrix_coop_sync(data, frag, ldm, waveIndex, waveCount);
    }

} // namespace rocwmma

#endif // ROCWMMA_COOP_API_IMPL_HPP
