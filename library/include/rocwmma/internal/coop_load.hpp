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

        template <uint32_t WaveCount, uint32_t SplitCount>
        ROCWMMA_DEVICE static inline void exec(typename Traits::OutputT& data,
                                               DataT const*              dataPtr,
                                               uint32_t                  ldm,
                                               uint32_t                  waveIndex)
        {
            // Ensure that splitCount doesn't exceed our maximum
            constexpr auto splitCount = std::min(SplitCount, (uint32_t)Traits::MaxSplit);

            // For the cases where there are more waves than splits.
            if(waveIndex >= splitCount)
                return;

            // Calculate the number of 'work items' for the current wave,
            // as well as the IOCount per work item.
            // NOTE: If there are in fact more waves than work items, make sure there
            // is at least one work item per wave. Waves that can't contribute will be
            // filtered out by the above check.
            constexpr auto workItemCount   = std::max(splitCount / WaveCount, 1u);
            constexpr auto workItemIOCount = IOTraits::IOCount / splitCount;

            // Calculate the current wave's starting IO iterator index for the first work item.
            // Calculate the IO offset between work items for the current wave.
            auto& reducedFt = reinterpret_cast<typename LoadVecTraits::template VecT<
                DataT,
                workItemCount * workItemIOCount * LoadVecTraits::size()>&>(data);
            auto  ioIter    = makeVectorIterator<LoadVecTraits::size()>(reducedFt).begin();

            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Iterate through the work items for this wave only.
            // Both loops may get unrolled if splitCount and waveCount are known at compile time.
#pragma unroll
            for(uint32_t i = 0; i < workItemCount; i++)
            {
                auto cumOffset = (i * WaveCount + waveIndex) * workItemIOCount;
#pragma unroll
                for(uint32_t j = 0; j < workItemIOCount; ++j)
                {
                    Traits::Loader::exec(
                        *ioIter,
                        dataPtr,
                        DataLayout::fromMatrixCoord(
                            baseOffset + MatrixLayout::cumulativeOffset(cumOffset++), ldm));
                    ioIter++;
                }
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_LOAD_HPP
