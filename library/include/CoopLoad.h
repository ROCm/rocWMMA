/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef WMMA_COOP_LOAD_H
#define WMMA_COOP_LOAD_H

#include "IOTraits.h"
#include "Layout.h"
#include "OpaqueLoad.h"
#include "Types.h"
#include "Utils.h"

namespace rocwmma
{

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataMapper,
              class MatrixMapper,
              uint32_t VectorWidth,
              uint32_t SpCount = 0>
    struct CooperativeLoad
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                SplitCount   = SpCount,
                SplitIOCount = IOTraits::IOCount / SplitCount
            };

            // Load implementation
            // Iteratively loads the entire block
            using Loader = detail::amdgcn_opaque_load<DataT, VectorWidth>;
            using LoadT  = typename Loader::LoadT;

            // Block output vector
            using OutputT = VecT<DataT, IOTraits::UnpackedSize>;

            static_assert(SplitCount > 0 && SplitCount <= IOTraits::IOCount,
                          "Invalid SplitCount range");
            static_assert(IOTraits::IOCount % SplitCount == 0,
                          "IOCount must be divisible by SplitCount");
            static_assert(OutputT::size() % SplitCount == 0,
                          "Register count not divisible by SplitCount");
            static_assert(OutputT::size() / SplitCount >= 1, "Partial registers not supported");
        };

        __device__ static inline void exec(typename Traits::OutputT& output,
                                           DataT const*              loadPtr,
                                           uint32_t                  ldm,
                                           uint32_t                  waveIndex,
                                           uint32_t                  waveCount)
        {
            using Loader = typename Traits::Loader;

            // For the cases where there are more groups than splits.
            if(waveIndex >= Traits::SplitCount)
                return;

            // Align threads to starting positions
            auto baseOffset = MatrixMapper::baseOffset();

            // Break down block into iterable loads
            auto splitIter = output.template begin<Traits::LoadT::size()>();

#pragma unroll
            for(uint32_t i = 0; i < Traits::SplitCount; ++i)
            {
                if(i % waveCount == waveIndex)
                {
                    auto ioIter = splitIter;
#pragma unroll
                    for(uint32_t j = 0; j < Traits::SplitIOCount; ++j)
                    {
                        *ioIter = *Loader::exec(
                            loadPtr,
                            DataMapper::fromMatrixCoord(
                                baseOffset + MatrixMapper::cumulativeOffset(ioIter.index()), ldm));
                        ioIter++;
                    }
                }
                splitIter += Traits::SplitIOCount;
            }
        }
    };

    // Wrapper for runtime wave count
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataMapper,
              class MatrixMapper,
              uint32_t VectorWidth>
    struct CooperativeLoad<BlockDim, BlockK, DataT, DataMapper, MatrixMapper, VectorWidth, 0>
    {
        template <uint32_t SplitCount>
        using CoopLoad = CooperativeLoad<BlockDim,
                                         BlockK,
                                         DataT,
                                         DataMapper,
                                         MatrixMapper,
                                         VectorWidth,
                                         SplitCount>;

        struct Traits
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

            enum : uint32_t
            {
                MaxSplit = IOTraits::IOCount
            };

            // All loads will have the same result type
            using OutputT = typename CoopLoad<1>::Traits::OutputT;
        };

        /*
    * While we try to do the runtime dispatching, we need to make sure that we only
    * instantiate splitting functions that make sense. The maximum possible split is the
    * same value as IOCount, which for now we will limit to 8.
    *
    * Note: The additional template parameter OutgoingT sets us up for proper forwarding
    * technique while allowing us to use it as the dependent parameter to exploit SFINAE
    * and hide instantiations that would be otherwise not compileable.
    */

        // IOCount of 8+ can potentially split work between 8 waves
        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit >= 64,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 64)
            {
                CoopLoad<64>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 32)
            {
                CoopLoad<32>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 16)
            {
                CoopLoad<16>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 8)
            {
                CoopLoad<8>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 4)
            {
                CoopLoad<4>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 32,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 32)
            {
                CoopLoad<32>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 16)
            {
                CoopLoad<16>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 8)
            {
                CoopLoad<8>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 4)
            {
                CoopLoad<4>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 16,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 16)
            {
                CoopLoad<16>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 8)
            {
                CoopLoad<8>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 4)
            {
                CoopLoad<4>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 8,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 8)
            {
                CoopLoad<8>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 4)
            {
                CoopLoad<4>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 4,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 4)
            {
                CoopLoad<4>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 2,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 2)
            {
                CoopLoad<2>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else if(splitCount == 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }

        template <typename OutgoingT,
                  typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                       typename std::decay<OutgoingT>::type>::value
                                              && Traits::MaxSplit == 1,
                                          int>::type
                  = 0>
        __device__ static inline void exec(OutgoingT&&  output,
                                           DataT const* dataPtr,
                                           uint32_t     ldm,
                                           uint32_t     waveIndex,
                                           uint32_t     waveCount,
                                           uint32_t     splitCount)
        {
            if(splitCount >= 1)
            {
                CoopLoad<1>::exec(
                    std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
            }
            else
            {
                assert(0 && "Unsupported split count. Try reducing workgroup waves.");
            }
        }
    };

} // namespace rocwmma

#endif // WMMA_COOP_LOAD_H
