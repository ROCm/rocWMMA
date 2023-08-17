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
#ifndef ROCWMMA_COOP_STORE_HPP
#define ROCWMMA_COOP_STORE_HPP

#include "io_traits.hpp"
#include "layout.hpp"
#include "opaque_store.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace rocwmma
{

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
              uint32_t VectorWidth>
    struct CooperativeStore
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                MaxSplit = IOTraits::IOCount
            };

            // Store implementation
            // Iteratively stores the entire block
            using Storer = detail::amdgcn_opaque_store<DataT, VectorWidth>;
            using StoreT = typename Storer::StoreT;

            // Block input vector
            using InputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using StoreVecTraits = VecTraits<typename Traits::StoreT>;

        ROCWMMA_DEVICE static inline void exec(DataT*                         dataPtr,
                                               typename Traits::InputT const& data,
                                               uint32_t                       ldm,
                                               uint32_t                       waveIndex,
                                               uint32_t                       waveCount,
                                               uint32_t                       splitCount)
        {
            // Ensure that splitCount doesn't exceed our maximum
            splitCount = std::min(splitCount, (uint32_t)Traits::MaxSplit);

            // For the cases where there are more waves than splits.
            if(waveIndex >= splitCount)
                return;

            // Calculate the number of 'work items' for the current wave,
            // as well as the IOCount per work item.
            // NOTE: If there are in fact more waves than work items, make sure there
            // is at least one work item per wave. Waves that can't contribute will be
            // filtered out by the above check.
            auto workItemCount   = std::max(splitCount / waveCount, 1u);
            auto workItemIOCount = IOTraits::IOCount / splitCount;

            // Calculate the current wave's starting IO iterator index for the first work item.
            // Calculate the IO offset between work items for the current wave.
            auto ioIter
                = makeVectorIterator<StoreVecTraits::size()>(data).it(waveIndex * workItemIOCount);
            auto workItemIOInc = waveCount * workItemIOCount;

            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Iterate through the work items for this wave only
            // Both loops may get unrolled if splitCount and waveCount are known at compile time.
            for(uint32_t i = 0; i < workItemCount; i++)
            {
                auto workItemIOIter = ioIter;
                for(uint32_t j = 0; j < workItemIOCount; ++j)
                {
                    Traits::Storer::exec(
                        dataPtr,
                        *workItemIOIter,
                        DataLayout::fromMatrixCoord(
                            baseOffset + MatrixLayout::cumulativeOffset(workItemIOIter.index()),
                            ldm));
                    workItemIOIter++;
                }
                ioIter += waveCount * workItemIOCount;
            }
        }

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <std::size_t Depth = 0,
                  typename Iterator,
                  typename StrideCounts,
                  typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(DataT*         dataPtr,
                                                       Iterator&      in,
                                                       uint32_t       ldm,
                                                       StrideCounts&& strideCounts,
                                                       Strides2d&&    strides2d)
        {
            auto strideOffset = DataLayout::fromMatrixCoord(std::get<Depth>(strides2d), ldm);
            auto strideCount  = std::get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (std::tuple_size<std::decay_t<StrideCounts>>::value - 1u))
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Storer::exec(dataPtr, *in);
                    dataPtr += strideOffset;
                    in++;
                }
            }
            // Recurse to the next nested layer
            else
            {
                if(strideCount > 0)
                {
#pragma unroll
                    for(int i = 0; i < strideCount; i++)
                    {
                        unroll_right<Depth + 1>(dataPtr, in, ldm, strideCounts, strides2d);
                        dataPtr += strideOffset;
                        //in++;
                    }
                }
                else
                {
                    unroll_right<Depth + 1>(dataPtr, in, ldm, strideCounts, strides2d);
                }
            }
        }

        constexpr static uint32_t calcMaxWaves(uint32_t workItems, uint32_t waveCount)
        {
            return (workItems % waveCount == 0 ? waveCount
                                               : calcMaxWaves(workItems, waveCount / 2));
        };

        template <uint32_t WaveCount, uint32_t SplitCount>
        ROCWMMA_DEVICE static inline void exec(DataT*                         dataPtr,
                                               typename Traits::InputT const& data,
                                               uint32_t                       ldm,
                                               uint32_t                       waveIndex)
        {
            if(waveIndex >= WaveCount)
                return;
            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Per-fragment work
            constexpr auto accum = [](auto... items) { return ((items == 0 ? 1u : items) * ...); };
            constexpr auto strideCounts   = MatrixLayout::strideCounts();
            constexpr auto strides        = MatrixLayout::strides();
            constexpr auto totalWorkItems = std::apply(accum, strideCounts);

            // Per-wave work.
            constexpr auto workItemsPerWave = std::max(totalWorkItems / WaveCount, 1u);
            auto&          reducedFt = reinterpret_cast<typename StoreVecTraits::template VecT<
                DataT,
                workItemsPerWave * StoreVecTraits::size()> const&>(data);
            auto           it = makeVectorIterator<StoreVecTraits::size()>(reducedFt).begin();

            // We know how much work each wave will do, however we need to divide up the strides
            // space evenly amongs the waves. Each wave is will at least fill MaxVW on its own, so
            // we can drop the first dimension and divide up the rest evenly.
            constexpr auto accum1          = [](auto... items) { return (items + ...); };
            constexpr auto strideCountsR   = pop_right(strideCounts);
            constexpr auto stridesR        = pop_right(strides);
            constexpr auto totalWorkItemsR = std::apply(accum, strideCountsR);
            constexpr auto waveCountAdjusted
                = calcMaxWaves((uint32_t)totalWorkItemsR, (uint32_t)WaveCount);

            if(waveIndex >= waveCountAdjusted)
                return;

            constexpr auto workItemsPerWaveR = totalWorkItemsR / waveCountAdjusted;
            constexpr auto waveStrides       = inflate_coord_left(workItemsPerWaveR, strideCountsR);

            auto currentWaveOffset = std::apply(
                accum1,
                inflate_coord_left(waveIndex * workItemsPerWaveR, strideCountsR) * stridesR);

            unroll_right(
                dataPtr + DataLayout::fromMatrixCoord(baseOffset + currentWaveOffset, ldm),
                it,
                ldm,
                std::tuple_cat(
                    waveStrides,
                    std::make_tuple(std::get<std::tuple_size<decltype(strideCounts)>::value - 1>(
                        strideCounts))),
                strides);
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_STORE_HPP
