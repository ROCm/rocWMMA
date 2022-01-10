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
#ifndef WMMA_COOP_STORE_H
#define WMMA_COOP_STORE_H

#include "IOTraits.h"
#include "Layout.h"
#include "OpaqueStore.h"
#include "Types.h"
#include "Utils.h"

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t VectorWidth,
          uint32_t SpCount = 0>
struct amdgcn_cooperative_store_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
    struct Traits
    {
        enum : uint32_t
        {
            SplitCount   = SpCount,
            SplitIOCount = IOTraits::IOCount / SplitCount
        };

        // Matrix-space thread offsets
        using MatrixLayout = StoreLayout<BlockDim, BlockK, DataT, DataLayout, VectorWidth>;
        using MappingUtil  = typename MatrixLayout::Traits::MappingUtil;

        // Store implementation
        // Iteratively stores the entire block
        using Storer = amdgcn_opaque_store<DataT, VectorWidth>;
        using StoreT = typename Storer::StoreT;

        // Block input vector
        using InputT = VecT<DataT, IOTraits::UnpackedSize>;

        static_assert(SplitCount > 0 && SplitCount <= IOTraits::IOCount,
                      "Invalid SplitCount range");
        static_assert(IOTraits::IOCount % SplitCount == 0,
                      "IOCount must be divisible by SplitCount");
        static_assert(InputT::size() % SplitCount == 0,
                      "Register count not divisible by SplitCount");
        static_assert(InputT::size() / SplitCount >= 1, "Partial registers not supported");
    };

    __device__ static inline void exec(DataT*                         storePtr,
                                       typename Traits::InputT const& input,
                                       uint32_t                       ldm,
                                       uint32_t                       waveIndex,
                                       uint32_t                       waveCount)
    {
        using Storer       = typename Traits::Storer;
        using MatrixLayout = typename Traits::MatrixLayout;
        using MappingUtil  = typename Traits::MappingUtil;

        // For the cases where there are more waves than splits.
        if(waveIndex >= Traits::SplitCount)
            return;

        // Align threads to starting positions
        auto baseOffset = MatrixLayout::baseOffset();

        // Break down block into iterable stores
        auto splitIter = input.template begin<Traits::StoreT::size()>();

#pragma unroll
        for(uint32_t i = 0; i < Traits::SplitCount; ++i)
        {
            if(i % waveCount == waveIndex)
            {
                auto ioIter = splitIter;
#pragma unroll
                for(uint32_t j = 0; j < Traits::SplitIOCount; ++j)
                {
                    Storer::exec(
                        storePtr,
                        *ioIter,
                        MappingUtil::dataOffset(
                            ldm, baseOffset + MatrixLayout::cumulativeOffset(ioIter.index())));
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
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t VectorWidth>
struct amdgcn_cooperative_store_DxK<BlockDim,
                                    BlockK,
                                    DataT,
                                    DataLayout,
                                    StoreLayout,
                                    VectorWidth,
                                    0>
{
    template <uint32_t SplitCount>
    using CooperativeStore = amdgcn_cooperative_store_DxK<BlockDim,
                                                          BlockK,
                                                          DataT,
                                                          DataLayout,
                                                          StoreLayout,
                                                          VectorWidth,
                                                          SplitCount>;

    struct Traits
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;

        enum : uint32_t
        {
            MaxSplit = IOTraits::IOCount
        };

        // All stores will have the same input block type
        using InputT = typename CooperativeStore<1>::Traits::InputT;

        // Determine the allowable split count from the layout
        using StoreLayoutT = StoreLayout<BlockDim, BlockK, DataT, DataLayout, VectorWidth>;
    };

    /*
    * While we try to do the runtime dispatching, we need to make sure that we only
    * instantiate splitting functions that make sense. The maximum possible split is the
    * same value as IOCount, which for now we will limit to 8.
    *
    * Note: The additional template parameter IncomingT sets us up for proper forwarding
    * technique while allowing us to use it as the dependent parameter to exploit SFINAE
    * and hide instantiations that would be otherwise not compileable.
    */

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit >= 64,
                  int>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 64)
        {
            CooperativeStore<64>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 32)
        {
            CooperativeStore<32>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 16)
        {
            CooperativeStore<16>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 8)
        {
            CooperativeStore<8>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 32,
                  int>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 32)
        {
            CooperativeStore<32>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 16)
        {
            CooperativeStore<16>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 8)
        {
            CooperativeStore<8>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 16,
                  int>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 16)
        {
            CooperativeStore<16>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 8)
        {
            CooperativeStore<8>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 8,
                  int>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 8)
        {
            CooperativeStore<8>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 4,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 2,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(splitCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && Traits::MaxSplit == 1,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(DataT*      dataPtr,
                                       IncomingT&& input,
                                       uint32_t    ldm,
                                       uint32_t    waveIndex,
                                       uint32_t    waveCount,
                                       uint32_t    splitCount)
    {
        if(splitCount >= 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }
};

#endif // WMMA_COOP_STORE_H
