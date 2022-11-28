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
#ifndef ROCWMMA_WMMA_HPP
#define ROCWMMA_WMMA_HPP

#include "vector.hpp"
#include "vector_iterator.hpp"

#include "wmma_impl.hpp"

namespace rocwmma
{
    // Wmma interface
    template <typename InputT,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename Enabler = void>
    struct Wmma : public detail::amdgcn_wmma<InputT, ComputeT, BlockM, BlockN>
    {
    };

    // Unlock the WMMA for NAVI cards
    // Supported Input/Compute types:
    // float16_t / float16_t
    // float16_t / float32_t
    // hfloat16_t / float16_t
    // hfloat16_t / float32_t
    // bfloat16_t / bfloat16_t
    // bfloat16_t / float32_t
    // int8_t / int32_t
    // Supported block sizes (M, N) = 16
    template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct Wmma<
        InputT,
        ComputeT,
        BlockM,
        BlockN,
        BlockK,
        typename std::enable_if<
            ROCWMMA_ARCH_NAVI // NAVI only
            && ((std::is_same<InputT, float16_t>::value && std::is_same<ComputeT, float16_t>::value)
                || (std::is_same<InputT, float16_t>::value
                    && std::is_same<ComputeT, float32_t>::value)
                || (std::is_same<InputT, hfloat16_t>::value
                    && std::is_same<ComputeT, hfloat16_t>::value)
                || (std::is_same<InputT, hfloat16_t>::value
                    && std::is_same<ComputeT, float32_t>::value)
                || (std::is_same<InputT, bfloat16_t>::value
                    && std::is_same<ComputeT, bfloat16_t>::value)
                || (std::is_same<InputT, bfloat16_t>::value
                    && std::is_same<ComputeT, float32_t>::value)
                || (std::is_same<InputT, int8_t>::value && std::is_same<ComputeT, int32_t>::value))
            && (BlockM == 16) && (BlockN == 16) && (BlockK >= 16) // 16 block size only
            >::type>
    {
        // Full-fragment IO traits
        using IOTraitsA   = IOTraits<BlockM, BlockK, InputT>;
        using IOTraitsB   = IOTraits<BlockK, BlockN, InputT>;
        using IOTraitsAcc = IOTraits<BlockM, BlockN, ComputeT>;

        // Functional
        using WMMA = detail::amdgcn_wmma<InputT, ComputeT, BlockM, BlockN>;

        // Per-WMMA iterative vector requirements
        using VecTraitsA = VecTraits<typename WMMA::Traits::ARegsT>;
        using VecTraitsB = VecTraits<typename WMMA::Traits::BRegsT>;
        using VecTraitsC = VecTraits<typename WMMA::Traits::CRegsT>;
        using VecTraitsD = VecTraits<typename WMMA::Traits::DRegsT>;

        struct Traits
        {
            enum : uint32_t
            {
                WmmaCount = BlockK / WMMA::Traits::KPerWmma,
                MinK      = WMMA::Traits::KPerWmma,

                // WMMA instructions need to duplicate inputs and therefore
                // must double fragment size.
                WmmaInputMultiplier = 2u
            };

            // Create full-fragment vector sizes
            using ARegsT = typename VecTraitsA::template VecT<typename VecTraitsA::DataT,
                                                              WmmaCount * VecTraitsA::size()
                                                                  / WmmaInputMultiplier>;
            using BRegsT = typename VecTraitsB::template VecT<typename VecTraitsB::DataT,
                                                              WmmaCount * VecTraitsB::size()
                                                                  / WmmaInputMultiplier>;
            using CRegsT = typename VecTraitsC::template VecT<>;
            using DRegsT = typename VecTraitsD::template VecT<>;

            // Create per-wmma fragment vector sizes
            using ARegsTPWmma =
                typename VecTraitsA::template VecT<typename VecTraitsA::DataT,
                                                   VecTraitsA::size() / WmmaInputMultiplier>;
            using BRegsTPWmma =
                typename VecTraitsB::template VecT<typename VecTraitsB::DataT,
                                                   VecTraitsB::size() / WmmaInputMultiplier>;

            // Sanity checks
            static_assert(BlockK >= MinK, "BlockK is not a minimum of MinK");
            static_assert(BlockK % MinK == 0, "BlockK is not a multiple of MinK");

            // A / B  and C / D types must match
            static_assert(
                std::is_same<typename VecTraitsA::DataT, typename VecTraitsB::DataT>::value,
                "A and B registers must be of same type");
            static_assert(
                std::is_same<typename VecTraitsC::DataT, typename VecTraitsD::DataT>::value,
                "C and D registers must be of same type");

            // Full fragment counts must match packed IO counts
            // WMMA expects packed elements
            static_assert(VecTraits<ARegsT>::size() == IOTraitsA::PackedSize,
                          "Unexpected packed vector size for A");
            static_assert(VecTraits<BRegsT>::size() == IOTraitsB::PackedSize,
                          "Unexpected packed vector size for B");
            static_assert(VecTraits<CRegsT>::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for C");
            static_assert(VecTraits<DRegsT>::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for D");
        };

        __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                           typename Traits::BRegsT const& regsB,
                                           typename Traits::CRegsT const& regsC) ->
            typename Traits::DRegsT
        {
            typename Traits::DRegsT result = regsC;

            // Iterate over WMMA input requirements
            auto aIt = makeVectorIterator<VecTraitsA::size() / Traits::WmmaInputMultiplier>(regsA)
                           .begin();
            auto bIt = makeVectorIterator<VecTraitsB::size() / Traits::WmmaInputMultiplier>(regsB)
                           .begin();

            // Accumulate over WMMA count
#pragma unroll
            for(unsigned i = 0; i < Traits::WmmaCount; i++)
            {
                //Duplicate the uppper/lower input for register A
                typename Traits::ARegsTPWmma regsAUpper(*aIt);
                typename Traits::ARegsTPWmma regsALower(*aIt);

                //Create and initialize Wmma input A register
                typename WMMA::Traits::ARegsT regsA_Wmma(InputT(0));

                //Iterators for upper half and lower half of wmma registers
                auto upperSrcAIt = makeVectorIterator(regsAUpper).begin();
                auto lowerSrcAIt = makeVectorIterator(regsALower).begin();

                static_assert(upperSrcAIt.range() == lowerSrcAIt.range(),
                              "Upper and Lower register size should be equal");

                for(int j = 0; j < upperSrcAIt.range(); j++)
                {
                    Permute<PermuteOps::BlockBCast16<0>>::exec(*lowerSrcAIt, detail::laneId());
                    Permute<PermuteOps::BlockBCast16<1>>::exec(*upperSrcAIt, detail::laneId());
                    upperSrcAIt++;
                    lowerSrcAIt++;
                }

                // update the src and dst iterators after data operation
                auto dstAIt = makeVectorIterator<VecTraitsA::size() / Traits::WmmaInputMultiplier>(
                                  regsA_Wmma)
                                  .begin();

                (*dstAIt) = regsAUpper;
                dstAIt++;
                (*dstAIt) = regsALower;

                //Duplicate the uppper/lower input for register B
                typename Traits::BRegsTPWmma regsBUpper(*bIt);
                typename Traits::BRegsTPWmma regsBLower(*bIt);

                //Create and initialize Wmma input B register
                typename WMMA::Traits::BRegsT regsB_Wmma(InputT(0));

                //Iterators for upper half and lower half of wmma registers
                auto upperSrcBIt = makeVectorIterator(regsBUpper).begin();
                auto lowerSrcBIt = makeVectorIterator(regsBLower).begin();

                static_assert(upperSrcBIt.range() == lowerSrcBIt.range(),
                              "Upper and Lower register size should be equal");

                for(int j = 0; j < upperSrcBIt.range(); j++)
                {
                    Permute<PermuteOps::BlockBCast16<0>>::exec(*lowerSrcBIt, detail::laneId());
                    Permute<PermuteOps::BlockBCast16<1>>::exec(*upperSrcBIt, detail::laneId());
                    upperSrcBIt++;
                    lowerSrcBIt++;
                }

                auto dstBIt = makeVectorIterator<VecTraitsB::size() / Traits::WmmaInputMultiplier>(
                                  regsB_Wmma)
                                  .begin();

                // update the src and dst iterators after data operation
                (*dstBIt) = regsBUpper;
                dstBIt++;
                (*dstBIt) = regsBLower;

                result = WMMA::exec(regsA_Wmma, regsB_Wmma, result);

                aIt++;
                bIt++;
            }
            return result;
        }
    };
} // namespace rocwmma

#endif // ROCWMMA_WMMA_HPP
