#ifndef WMMA_COOP_LOAD_H
#define WMMA_COOP_LOAD_H

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
          class LoadLayout,
          uint32_t ElementsPerThread,
          uint32_t SpCount = 0>
struct amdgcn_cooperative_load_DxK
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

        // Each cooperative wave will load a piece of the final output
        using OutputT = VecT<DataT, IOTraits::UnpackedSize>;
        static_assert(OutputT::size() % SplitCount == 0,
                      "Register count not divisible by SplitCount");

        // Partial loader
        using Loader = amdgcn_opaque_load_DxK<BlockDim,
                                              BlockK / Traits::SplitCount,
                                              DataT,
                                              DataLayout,
                                              LoadLayout,
                                              ElementsPerThread>;

        using LoaderLayout = LoadLayout<BlockDim,
                                        BlockK / Traits::SplitCount,
                                        DataT,
                                        DataLayout,
                                        ElementsPerThread>;
    };

    __device__ static inline void exec(typename Traits::OutputT& output,
                                       DataT const*              dataPtr,
                                       uint32_t                  ldm,
                                       uint32_t                  waveIndex,
                                       uint32_t                  waveCount)
    {
        using Loader       = typename Traits::Loader;
        using LoaderLayout = typename Traits::LoaderLayout;

        // For the cases where there are more groups than splits.
        waveIndex = waveIndex % Traits::SplitCount;

        // Determine offset for the current wave, and
        // emplace the partial load into the output.
        auto loadOffset = LoaderLayout::dataBlockOffset(ldm) * waveIndex;

        auto it = output.template begin<Traits::OutputT::size() / Traits::SplitCount>();

        // This next part is important for optimization.
        // The goal is to do a partial load and write only one part of the fragment.
        // However, from the compiler perspective it does not optimize assembly
        // correctly if we load only the part of the fragment without referencing the rest of the fragment.
        // For example, we could easily do the following:
        //
        //  auto it =
        //      typename Traits::OutputT::template Iterator<Traits::OutputT::size() / Traits::SplitCount>(
        //        input, waveIndex);
        //  *it = *Loader::exec(dataPtr + loadOffset, ldm);
        //
        // However we didn't touch the rest of the fragment, so the compiler may decide to dump the fragment
        // to memory as it will treat this as a sub-array access.
        //
        // To keep it in registers, iterate over the entire fragment and load only when we reach our intended
        // offset.

#pragma unroll
        for(uint32_t i = 0; i < Traits::SplitCount; ++i)
        {
            if(i == waveIndex)
            {
                *it = *Loader::exec(dataPtr + loadOffset, ldm);
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
          class LoadLayout,
          uint32_t ElementsPerThread>
struct amdgcn_cooperative_load_DxK<BlockDim,
                                   BlockK,
                                   DataT,
                                   DataLayout,
                                   LoadLayout,
                                   ElementsPerThread,
                                   0>
{
    template <uint32_t SplitCount>
    using CooperativeLoad = amdgcn_cooperative_load_DxK<BlockDim,
                                                        BlockK,
                                                        DataT,
                                                        DataLayout,
                                                        LoadLayout,
                                                        ElementsPerThread,
                                                        SplitCount>;

    // All loads will have the same result type
    struct Traits
    {
        using OutputT = typename CooperativeLoad<1>::Traits::OutputT;
    };

    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    /*
    * While we try to do the runtime dispatching, we need to make sure that we only
    * instantiate splitting functions that make sense. The maximum possible split is 8
    * but this only makes sense if the packed IOCount is divisible by 8. Otherwise we
    * will have an explosion of static asserts from the IOTraits class during compile time.
    *
    * Note: The additional template parameter OutgoingT sets us up for proper forwarding
    * technique while allowing us to use it as the dependent parameter to exploit SFINAE
    * and hide instantiations that would be otherwise not compileable.
    */

    // IOCount of 8+ can potentially split work between 8 waves
    template <typename OutgoingT,
              typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                   typename std::decay<OutgoingT>::type>::value
                                          && IOTraits::IOCount / PackTraits<DataT>::PackRatio >= 8,
                                      int>::type
              = 0>
    __device__ static inline void exec(OutgoingT&&  output,
                                       DataT const* dataPtr,
                                       uint32_t     ldm,
                                       uint32_t     waveIndex,
                                       uint32_t     waveCount)
    {
        if(waveCount >= 8)
        {
            CooperativeLoad<8>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 4)
        {
            CooperativeLoad<4>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 2)
        {
            CooperativeLoad<2>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
        {
            CooperativeLoad<1>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // IOCount of 8+ can potentially split work between 8 waves
    template <typename OutgoingT,
              typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                   typename std::decay<OutgoingT>::type>::value
                                          && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 4,
                                      int>::type
              = 0>
    __device__ static inline void exec(OutgoingT&&  output,
                                       DataT const* dataPtr,
                                       uint32_t     ldm,
                                       uint32_t     waveIndex,
                                       uint32_t     waveCount)
    {
        if(waveCount >= 4)
        {
            CooperativeLoad<4>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 2)
        {
            CooperativeLoad<2>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
        {
            CooperativeLoad<1>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // IOCount of 4 can potentially split work between 4 waves
    template <typename OutgoingT,
              typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                   typename std::decay<OutgoingT>::type>::value
                                          && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 2,
                                      int>::type
              = 0>
    __device__ static inline void exec(OutgoingT&&  output,
                                       DataT const* dataPtr,
                                       uint32_t     ldm,
                                       uint32_t     waveIndex,
                                       uint32_t     waveCount)
    {
        if(waveCount >= 2)
        {
            CooperativeLoad<2>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else if(waveCount == 1)
        {
            CooperativeLoad<1>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // IOCount of 2 can potentially split work between 2 waves
    template <typename OutgoingT,
              typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                   typename std::decay<OutgoingT>::type>::value
                                          && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 1,
                                      int>::type
              = 0>
    __device__ static inline void exec(OutgoingT&&  output,
                                       DataT const* dataPtr,
                                       uint32_t     ldm,
                                       uint32_t     waveIndex,
                                       uint32_t     waveCount)
    {
        if(waveCount >= 1)
        {
            CooperativeLoad<1>::exec(
                std::forward<OutgoingT>(output), dataPtr, ldm, waveIndex, waveCount);
        }
        else
        {
            assert(0 && "Unsupported split count. Try reducing workgroup waves.");
        }
    }

    // Intentionally left undefined. If you have 0 IO count, there are other problems!
    template <typename OutgoingT,
              typename std::enable_if<std::is_same<typename Traits::OutputT,
                                                   typename std::decay<OutgoingT>::type>::value
                                          && IOTraits::IOCount / PackTraits<DataT>::PackRatio == 0,
                                      int>::type
              = 0>
    __device__ static inline void exec(OutgoingT&&  output,
                                       DataT const* dataPtr,
                                       uint32_t     ldm,
                                       uint32_t     waveIndex,
                                       uint32_t     waveCount);
};

#endif // WMMA_COOP_LOAD_H
