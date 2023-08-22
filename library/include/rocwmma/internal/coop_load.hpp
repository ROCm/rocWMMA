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
#ifndef ROCWMMA_COOP_LOAD_HPP
#define ROCWMMA_COOP_LOAD_HPP

#include "io_traits.hpp"
#include "layout.hpp"
#include "opaque_load.hpp"
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
    struct CooperativeLoad
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                MaxSplit = IOTraits::IOCount
            };

            // Load implementation
            using Loader = detail::amdgcn_opaque_load<DataT, VectorWidth>;
            using LoadT  = typename Loader::LoadT;

            // Block output vector
            using OutputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using LoadVecTraits = VecTraits<typename Traits::LoadT>;

        ROCWMMA_DEVICE static inline void exec(typename Traits::OutputT& data,
                                               DataT const*              dataPtr,
                                               uint32_t                  ldm,
                                               uint32_t                  waveIndex,
                                               uint32_t                  waveCount,
                                               uint32_t                  splitCount)
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
                = makeVectorIterator<LoadVecTraits::size()>(data).it(waveIndex * workItemIOCount);
            auto workItemIOInc = waveCount * workItemIOCount;

            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Iterate through the work items for this wave only.
            // Both loops may get unrolled if splitCount and waveCount are known at compile time.
            for(uint32_t i = 0; i < workItemCount; i++)
            {
                auto workItemIOIter = ioIter;
                for(uint32_t j = 0; j < workItemIOCount; ++j)
                {
                    Traits::Loader::exec(
                        *workItemIOIter,
                        dataPtr,
                        DataLayout::fromMatrixCoord(
                            baseOffset + MatrixLayout::cumulativeOffset(workItemIOIter.index()),
                            ldm));
                    workItemIOIter++;
                }
                ioIter += workItemIOInc;
            }
        }

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <std::size_t Depth = 0,
                  typename Iterator,
                  typename StrideCounts,
                  typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(Iterator&      out,
                                                       DataT const*   dataPtr,
                                                       uint32_t       ldm,
                                                       StrideCounts&& strideCounts,
                                                       Strides2d&&    strides2d)
        {
            static_assert(std::tuple_size<std::decay_t<StrideCounts>>::value
                              == std::tuple_size<std::decay_t<Strides2d>>::value,
                          "Mismatched size");
            auto strideOffset = DataLayout::fromMatrixCoord(std::get<Depth>(strides2d), ldm);
            auto strideCount  = std::get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (std::tuple_size<std::decay_t<StrideCounts>>::value - 1u))
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Loader::exec(*out, dataPtr);
                    dataPtr += strideOffset;
                    out++;
                }
            }
            // Recurse to the next nested layer
            else
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_right<Depth + 1>(out, dataPtr, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                    //out++;
                }
            }
        }

        constexpr static uint32_t calcMaxWaves(uint32_t workItems, uint32_t waveCount)
        {
            return (workItems % waveCount == 0 ? waveCount
                                               : calcMaxWaves(workItems, waveCount / 2));
        };

        template <uint32_t WaveCount, uint32_t SplitCount>
        ROCWMMA_DEVICE static inline void exec(typename Traits::OutputT& data,
                                               DataT const*              dataPtr,
                                               uint32_t                  ldm,
                                               uint32_t                  waveIndex)
        {
            // Full fragment work
            constexpr auto strideSpace = MatrixLayout::strideCounts();
            constexpr auto strides     = MatrixLayout::strides();

            // Drop the VW strides for splitting (reduced stride space).
            constexpr auto strideSpaceR = pop_right(strideSpace);
            constexpr auto stridesR     = pop_right(strides);
            constexpr auto totalWorkItems
                = flatten_coord_left((strideSpaceR - 1u), strideSpaceR) + 1u;

            // Determine max waves possible.
            constexpr auto waveCountAdjusted
                = calcMaxWaves((uint32_t)totalWorkItems, (uint32_t)WaveCount);

            // Ensure that we propagate the corrected wave count
            if constexpr(WaveCount != waveCountAdjusted)
            {
                exec<waveCountAdjusted, SplitCount>(data, dataPtr, ldm, waveIndex);
            }
            // Wave count is acceptable
            else
            {
                // Don't go beyond the scope of work
                waveIndex %= WaveCount;

                // Split the reduced stride space.
                constexpr auto workItemsPerWave = std::max(totalWorkItems / WaveCount, 1u);
                constexpr auto strideSpaceS
                    = inflate_coord_left(workItemsPerWave - 1u, strideSpaceR) + 1u;

                // Add back in the VW dimension, for the full stride
                // space of the current wave
                constexpr auto strideSpaceW
                    = std::tuple_cat(strideSpaceS, std::make_tuple(get_last(strideSpace)));

                // Alias the original frag due to smaller split size
                auto& dataR
                    = (typename LoadVecTraits::
                           template VecT<DataT, workItemsPerWave * LoadVecTraits::size()>&)(data);
                auto it = makeVectorIterator<LoadVecTraits::size()>(dataR).begin();

                // Align threads to starting matrix offset coordinates
                auto baseOffset = MatrixLayout::baseOffset();

                // Find current wave offset
                constexpr auto sum               = [](auto... items) { return (items + ...); };
                auto           currentWaveOffset = std::apply(
                    sum, inflate_coord_left(waveIndex * workItemsPerWave, strideSpaceR) * stridesR);

                // if(threadIdx.x % 64 == 0)
                // {

                //     printf("(%d) BlockDim, BlockK: (%d, %d)\n", waveIndex, BlockDim, BlockK);
                //     printf("(%d) Original strideCounts: (%d, %d, %d)\n", waveIndex, std::get<0>(strideSpace), std::get<1>(strideSpace), std::get<2>(strideSpace));
                //     printf("(%d) Original strides: (%d, %d), (%d, %d), (%d, %d)\n", waveIndex, get<0>(std::get<0>(strides)), get<1>(std::get<0>(strides)),
                //                                                                             get<0>(std::get<1>(strides)), get<1>(std::get<1>(strides)),
                //                                                                             get<0>(std::get<2>(strides)), get<1>(std::get<2>(strides)));

                //     printf("(%d) WaveCount: (%d)\n", waveIndex, WaveCount);
                //     printf("(%d) workItemsPerWave (%d)\n", waveIndex, workItemsPerWave);
                //     printf("(%d) strideSpaceW: (%d, %d, %d)\n", waveIndex, std::get<0>(strideSpaceW), std::get<1>(strideSpaceW), std::get<2>(strideSpaceW));
                //     printf("(%d) currentOffset: (%d, %d)\n\n", waveIndex, get<0>(currentWaveOffset), get<1>(currentWaveOffset));

                // }

                unroll_right(it,
                             dataPtr
                                 + DataLayout::fromMatrixCoord(baseOffset + currentWaveOffset, ldm),
                             ldm,
                             strideSpaceW,
                             strides);
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_LOAD_HPP
