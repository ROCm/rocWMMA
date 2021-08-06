#ifndef WMMA_COOP_STORE_H
#define WMMA_COOP_STORE_H

#include "IOTraits.h"
#include "Layout.h"
#include "MappingUtil.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Types.h"

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t ElementsPerThread,
          uint32_t SpCount = 0>
struct amdgcn_cooperative_store_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT>;
    struct Traits
    {
        enum : uint32_t
        {
            SplitCount = SpCount,
        };

        static_assert(SplitCount > 0, "Split count must be greater than 0");
        static_assert(BlockK % SplitCount == 0, "BlockK size is not divisible by SplitCount");

        // Same register count for each split
        using InputT = VecT<DataT, IOTraits::UnpackedSize>;
        static_assert(InputT::size() % SplitCount == 0,
                      "Register count not divisible by SplitCount");
    };

    __device__ static inline void exec(DataT*                         dataPtr,
                                       typename Traits::InputT const& input,
                                       uint32_t                       ldm,
                                       uint32_t                       waveIndex,
                                       uint32_t                       waveCount)
    {
        // For the cases where there are more groups than splits.
        waveIndex = waveIndex % Traits::SplitCount;

        // Base address is the same, and split load by (SplitCount).
        // Multiply the gridId by the split load count to get iterative offset per wave.
        using Storer = amdgcn_opaque_store_DxK<BlockDim,
                                               BlockK / Traits::SplitCount,
                                               DataT,
                                               DataLayout,
                                               StoreLayout,
                                               ElementsPerThread>;

        using StoreLayoutT = StoreLayout<BlockDim,
                                         BlockK / Traits::SplitCount,
                                         DataT,
                                         DataLayout,
                                         ElementsPerThread>;

        auto storeOffset = StoreLayoutT::dataBlockOffset(ldm) * waveIndex;

        auto it = input.template begin<Traits::InputT::size() / Traits::SplitCount>();

// This next part is important for optimization.
// The goal is to do a partial store and write only one part of the fragment.
// However, from the compiler perspective it does not optimize assembly
// correctly if we directly write only the middle of the fragment (e.g. not the first regs)
// without referencing the rest of the fragment.
// For example, we could easily do the following:
//
//  auto it =
//      typename Traits::InputT::template Iterator<Traits::InputT::size() / Traits::SplitCount>(
//        input, waveIndex);
//  Storer::exec(dataPtr + storeOffset, *it, ldm);
//
// However we didn't touch the rest of the fragment, so the compiler may decide to dump the fragment
// to memory as it will treat this as a sub-array access.
//
// To keep it in registers, iterate over the entire fragment and store only when we reach our intended
// offset.
#pragma unroll
        for(uint32_t i = 0; i < Traits::SplitCount; ++i)
        {
            if(i == waveIndex)
            {
                Storer::exec(dataPtr + storeOffset, *it, ldm);
            }
            it++;
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
          uint32_t ElementsPerThread>
struct amdgcn_cooperative_store_DxK<BlockDim,
                                    BlockK,
                                    DataT,
                                    DataLayout,
                                    StoreLayout,
                                    ElementsPerThread,
                                    0>
{
    template <uint32_t SplitCount>
    using CooperativeStore = amdgcn_cooperative_store_DxK<BlockDim,
                                                          BlockK,
                                                          DataT,
                                                          DataLayout,
                                                          StoreLayout,
                                                          ElementsPerThread,
                                                          SplitCount>;

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    // All stores will have the same input type
    struct Traits
    {
        using InputT = typename CooperativeStore<1>::Traits::InputT;
    };

    /*
    * While we try to do the runtime dispatching, we need to make sure that we only
    * instantiate splitting functions that make sense. The maximum possible split is 8
    * but this only makes sense if the packed IOCount is divisible by 8. Otherwise we
    * will have an explosion of static asserts from the IOTraits class during compile time.
    *
    * Note: The additional template parameter IncomingT sets us up for proper forwarding
    * technique while allowing us to use it as the dependent parameter to exploit SFINAE
    * and hide instantiations that would be otherwise not compileable.
    */

    //IOCount of 8+ can potentially split work between 8 waves
    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && IOTraits::IOCount / PackTraits<DataT>::PackRatio >= 8,
                  int>::type
              = 0>
    __device__ static inline void exec(
        DataT* dataPtr, IncomingT&& input, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
    {
        if(waveCount >= 8)
        {
            CooperativeStore<8>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // IOCount of 4 can potentially split work between 4 waves
    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 4,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(
        DataT* dataPtr, IncomingT&& input, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
    {
        if(waveCount >= 4)
        {
            CooperativeStore<4>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // IOCount of 2 can potentially split work between 2 waves
    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 2,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(
        DataT* dataPtr, IncomingT&& input, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
    {
        if(waveCount >= 2)
        {
            CooperativeStore<2>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
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
                      && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 1,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(
        DataT* dataPtr, IncomingT&& input, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)
    {
        if(waveCount >= 1)
        {
            CooperativeStore<1>::exec(
                dataPtr, std::forward<IncomingT>(input), ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // Intentionally left undefined. If you have 0 IO count, there are other problems!
    template <typename IncomingT,
              typename std::enable_if<
                  std::is_same<typename Traits::InputT, typename std::decay<IncomingT>::type>::value
                      && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 0,
                  int32_t>::type
              = 0>
    __device__ static inline void exec(
        DataT* dataPtr, IncomingT&& input, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount);
};

#endif // WMMA_COOP_STORE_H
