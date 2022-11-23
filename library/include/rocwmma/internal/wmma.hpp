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

#include "permute.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN>
        struct amdgcn_wmma;

        template <>
        struct amdgcn_wmma<float16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<float16_t, float16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    regsA.data, regsB.data, regsC.data, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                    regsA.data, regsB.data, regsC.data, 1)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<int8_t, int32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegI32x4;
                using BRegsT = VRegI32x4;
                using CRegsT = AccRegI32x8;
                using DRegsT = AccRegI32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    1, regsA.data, 1, regsB.data, regsC.data, 1)};
                return result;
            }
        };
    } // namespace detail

    // Wmma class for unsupported types
    template <typename InputT,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              class enable = void>
    struct Wmma
    {
        template <typename TypeA, typename TypeB, typename TypeC>
        __device__ static inline auto
            exec(TypeA const& regsA, TypeB const& regsB, TypeC const& regsC)
        {
            return regsC;
        }
    };

    // Specified Wmma class for supported Block Sizes/ Input/ Compute Types
    template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct Wmma<
        InputT,
        ComputeT,
        BlockM,
        BlockN,
        BlockK,
        typename std::enable_if<
            ((std::is_same<InputT, float16_t>::value && std::is_same<ComputeT, float16_t>::value)
             || (std::is_same<InputT, float16_t>::value && std::is_same<ComputeT, float32_t>::value)
             || (std::is_same<InputT, bfloat16_t>::value && std::is_same<ComputeT, bfloat16_t>::value)
             || (std::is_same<InputT, bfloat16_t>::value && std::is_same<ComputeT, float32_t>::value)
             || (std::is_same<InputT, int8_t>::value && std::is_same<ComputeT, int32_t>::value))
            && (BlockM == 16) && (BlockN == 16) && (BlockK >= 16)>::type>
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
            };

            // Create full-fragment vector sizes
            using ARegsT = typename VecTraitsA::template VecT<typename VecTraitsA::DataT,
                                                              WmmaCount * VecTraitsA::size()
                                                                  / AMDGCN_CDNA_RDNA_WAVE_RATIO>;
            using BRegsT = typename VecTraitsB::template VecT<typename VecTraitsB::DataT,
                                                              WmmaCount * VecTraitsB::size()
                                                                  / AMDGCN_CDNA_RDNA_WAVE_RATIO>;
            using CRegsT = typename VecTraitsC::template VecT<>;
            using DRegsT = typename VecTraitsD::template VecT<>;

            // Create per-wmma fragment vector sizes
            using ARegsTPWmma =
                typename VecTraitsA::template VecT<typename VecTraitsA::DataT,
                                                   VecTraitsA::size()
                                                       / AMDGCN_CDNA_RDNA_WAVE_RATIO>;
            using BRegsTPWmma =
                typename VecTraitsB::template VecT<typename VecTraitsB::DataT,
                                                   VecTraitsB::size()
                                                       / AMDGCN_CDNA_RDNA_WAVE_RATIO>;

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
            auto aIt = makeVectorIterator<VecTraitsA::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(regsA)
                           .begin();
            auto bIt = makeVectorIterator<VecTraitsB::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(regsB)
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
                auto dstAIt = makeVectorIterator<VecTraitsA::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                                  regsA_Wmma)
                                  .begin();
                auto upperSrcAItFull
                    = makeVectorIterator<VecTraitsA::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                          regsAUpper)
                          .begin();
                auto lowerSrcAItFull
                    = makeVectorIterator<VecTraitsA::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                          regsALower)
                          .begin();

                (*dstAIt) = *upperSrcAItFull;
                dstAIt++;
                (*dstAIt) = *lowerSrcAItFull;

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

                auto dstBIt = makeVectorIterator<VecTraitsB::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                                  regsB_Wmma)
                                  .begin();
                auto upperSrcBItFull
                    = makeVectorIterator<VecTraitsB::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                          regsBUpper)
                          .begin();
                auto lowerSrcBItFull
                    = makeVectorIterator<VecTraitsB::size() / AMDGCN_CDNA_RDNA_WAVE_RATIO>(
                          regsBLower)
                          .begin();

                // update the src and dst iterators after data operation
                (*dstBIt) = *upperSrcBItFull;
                dstBIt++;
                (*dstBIt) = *lowerSrcBItFull;

                result = WMMA::exec(regsA_Wmma, regsB_Wmma, result);

                aIt++;
                bIt++;
            }
            return result;
        }
    };
} // namespace rocwmma

#endif // ROCWMMA_WMMA_HPP
