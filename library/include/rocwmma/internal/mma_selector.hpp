/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_MMA_SELECTOR_HPP
#define ROCWMMA_MMA_SELECTOR_HPP

#include "mma_traits.hpp"

namespace rocwmma
{

    namespace detail
    {
        // Inputs BlockM and BlockN are expected to be fixed (e.g., determined previously by other means).
        // This class will attempt to find the largest possible BlockK and map to a backend if it exists.
        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockKTest>
        struct MmaOpSelector
        {
        private:
            static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");

            // Candidate operation for the current params
            using CandidateOp = Mma_impl<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>;
            using CandidateTraits = MmaTraits<CandidateOp>;

        public:
            // If the candidate is supported (e.g., a backend implementation exists), then select it.
            // Otherwise, test another smaller BlockK. If no existing implementations, get a pass-through.
            using SelectedOp = conditional_t<CandidateTraits::is_supported,
                                             CandidateOp,
                                             typename MmaOpSelector<Mma_impl,
                                                                    InputTA,
                                                                    InputTB,
                                                                    ComputeT,
                                                                    BlockM,
                                                                    BlockN,
                                                                    BlockKTest / 2u>::SelectedOp>;
        };

        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN>
        struct MmaOpSelector<Mma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>
        {
            // Mma_impl will just be a pass-through if no instruction is found
            using SelectedOp = Mma_impl<InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>;
        };

        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  uint32_t FragM,
                  uint32_t FragN,
                  uint32_t FragK,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK>
        struct MmaOpSelectorTraits
        {
            // Given BlockMNK, find a suitable backend candidate and their traits
            using MmaOpSelector
                = MmaOpSelector<Mma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>;
            using SelectedOp  = typename MmaOpSelector::SelectedOp;
            using MmaOpTraits = MmaTraits<SelectedOp>;

            // Block counts for wave-tile mma
            constexpr static uint32_t BlocksM = FragM / BlockM;
            constexpr static uint32_t BlocksN = FragN / BlockN;
            constexpr static uint32_t BlocksK = FragK / BlockK;
            constexpr static uint32_t BlocksC = BlocksM * BlocksN;

            // Backend supported (exists)
            static constexpr bool MmaOpSupported = MmaOpTraits::is_supported;

            // Cost to run backend over wave-tile:
            // - invoke costs
            // - storage costs
            static constexpr uint32_t MmaCost
                = (BlockM / 16u * BlocksM) * (BlockN / 16u * BlocksN) * BlocksK;
            static constexpr uint32_t StorageCost = MmaOpTraits::BlockSizeA * BlocksM * BlocksK
                                                    + MmaOpTraits::BlockSizeB * BlocksN * BlocksK
                                                    + MmaOpTraits::BlockSizeC * BlocksC;

            // Optimization opportunity in interleaving if we have multiple blocks
            static constexpr uint32_t InterleaveOpportunity = BlocksM + BlocksN;
        };

        template <typename TraitsLhs, typename TraitsRhs, typename Enabler = void>
        struct MmaOpSelectorTraitsCompare;

        template <typename TraitsLhs, typename TraitsRhs>
        struct MmaOpSelectorTraitsCompare<
            TraitsLhs,
            TraitsRhs,
            enable_if_t<TraitsLhs::MmaOpSupported && TraitsRhs::MmaOpSupported>>
        {
            static constexpr auto TotalMmaCost = TraitsLhs::MmaCost + TraitsRhs::MmaCost;
            static constexpr auto TotalStorageCost
                = TraitsLhs::StorageCost + TraitsRhs::StorageCost;
            static constexpr auto TotalInterleaveOpportunity
                = TraitsLhs::InterleaveOpportunity + TraitsRhs::InterleaveOpportunity;

            // All weights must add to 1.0
            static constexpr auto MmaCostWeight               = 0.35f;
            static constexpr auto StorageCostWeight           = 0.35f;
            static constexpr auto InterleaveOpportunityWeight = 0.3f;

            // Costs are not favorable
            static constexpr auto MmaCostFactorLhs
                = (1.0f - ((float)TraitsLhs::MmaCost / (float)TotalMmaCost)) * MmaCostWeight;
            static constexpr auto StorageCostFactorLhs
                = (1.0f - ((float)TraitsLhs::StorageCost / (float)TotalStorageCost))
                  * StorageCostWeight;
            static constexpr auto MmaCostFactorRhs
                = (1.0f - ((float)TraitsRhs::MmaCost / (float)TotalMmaCost)) * MmaCostWeight;
            static constexpr auto StorageCostFactorRhs
                = (1.0f - ((float)TraitsRhs::StorageCost / (float)TotalStorageCost))
                  * StorageCostWeight;

            // Interleave opportunities are favorable
            static constexpr auto InterLeaveOpportunityFactorLhs
                = ((float)TraitsLhs::InterleaveOpportunity / (float)TotalInterleaveOpportunity)
                  * InterleaveOpportunityWeight;
            static constexpr auto InterLeaveOpportunityFactorRhs
                = ((float)TraitsRhs::InterleaveOpportunity / (float)TotalInterleaveOpportunity)
                  * InterleaveOpportunityWeight;

            // Sum weighted avg
            static constexpr auto WeightedFactorLhs
                = MmaCostFactorLhs + StorageCostFactorLhs + InterLeaveOpportunityFactorLhs;
            static constexpr auto WeightedFactorRhs
                = MmaCostFactorRhs + StorageCostFactorRhs + InterLeaveOpportunityFactorRhs;

            // The winner is of highest favor
            using Winner
                = conditional_t<WeightedFactorLhs >= WeightedFactorRhs, TraitsLhs, TraitsRhs>;
        };

        template <typename TraitsLhs, typename TraitsRhs>
        struct MmaOpSelectorTraitsCompare<
            TraitsLhs,
            TraitsRhs,
            enable_if_t<!TraitsLhs::MmaOpSupported && TraitsRhs::MmaOpSupported>>
        {
            // Automatic winner is supported op
            using Winner = TraitsRhs;
        };

        template <typename TraitsLhs, typename TraitsRhs>
        struct MmaOpSelectorTraitsCompare<
            TraitsLhs,
            TraitsRhs,
            enable_if_t<TraitsLhs::MmaOpSupported && !TraitsRhs::MmaOpSupported>>
        {
            // Automatic winner is supported op
            using Winner = TraitsLhs;
        };

        template <typename TraitsLhs, typename TraitsRhs>
        struct MmaOpSelectorTraitsCompare<
            TraitsLhs,
            TraitsRhs,
            enable_if_t<!TraitsLhs::MmaOpSupported && !TraitsRhs::MmaOpSupported>>
        {
            // Default: Lhs
            using Winner = TraitsLhs;
        };

        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  uint32_t FragM,
                  uint32_t FragN,
                  uint32_t FragK,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  typename Enabler = void>
        struct MmaSelector;

        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  uint32_t FragM,
                  uint32_t FragN,
                  uint32_t FragK,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT>
        struct MmaSelector<
            Mma_impl,
            FragM,
            FragN,
            FragK,
            InputTA,
            InputTB,
            ComputeT,
            enable_if_t<(bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED && (bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED
                        && !is_same_v<InputTA, void>>>
        {
        private:
            // Gather prospective backend information.
            // Invoke a tree-like selection between all the candidate backends.
            // For now, we can select between 16x16 and 32x32 block backends.
            using Mma16SelectorTraits = MmaOpSelectorTraits<Mma_impl,
                                                            FragM,
                                                            FragN,
                                                            FragK,
                                                            InputTA,
                                                            InputTB,
                                                            ComputeT,
                                                            16u,
                                                            16u,
                                                            FragK>;
            using Mma32SelectorTraits = MmaOpSelectorTraits<Mma_impl,
                                                            FragM,
                                                            FragN,
                                                            FragK,
                                                            InputTA,
                                                            InputTB,
                                                            ComputeT,
                                                            32u,
                                                            32u,
                                                            FragK>;

            // Compare and observe winner
            using MmaOpWinner = typename MmaOpSelectorTraitsCompare<Mma16SelectorTraits,
                                                                    Mma32SelectorTraits>::Winner;

        public:
            // Clamp to either FragM/N if they are smaller than the MmaDims
            static constexpr uint32_t MmaDimM = min(MmaOpWinner::MmaOpTraits::BlockM, FragM);
            static constexpr uint32_t MmaDimN = min(MmaOpWinner::MmaOpTraits::BlockN, FragN);

            using MmaOp = typename MmaOpWinner::SelectedOp;
        };

        template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t>
                  class Mma_impl,
                  uint32_t FragM,
                  uint32_t FragN,
                  uint32_t FragK,
                  typename InputTA,
                  typename InputTB,
                  typename ComputeT>
        struct MmaSelector<
            Mma_impl,
            FragM,
            FragN,
            FragK,
            InputTA,
            InputTB,
            ComputeT,
            enable_if_t<(bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED
                        && !(bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED && !is_same_v<InputTA, void>>>
        {
            // For now, only one choice for mma backend dims, 16x16
            using MmaOpWinner = MmaOpSelectorTraits<Mma_impl,
                                                    FragM,
                                                    FragN,
                                                    FragK,
                                                    InputTA,
                                                    InputTB,
                                                    ComputeT,
                                                    16u,
                                                    16u,
                                                    FragK>;

        public:
            // Clamp to either FragM/N if they are smaller than the MmaDims
            static constexpr uint32_t MmaDimM = min(MmaOpWinner::MmaOpTraits::BlockM, FragM);
            static constexpr uint32_t MmaDimN = min(MmaOpWinner::MmaOpTraits::BlockN, FragN);

            using MmaOp = typename MmaOpWinner::SelectedOp;
        };

    } // namespace detail

    template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t> class Mma_impl,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK>
    using MmaOpSelector
        = detail::MmaOpSelector<Mma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>;

    template <template <typename, typename, typename, uint32_t, uint32_t, uint32_t> class Mma_impl,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename InputTA,
              typename InputTB,
              typename ComputeT>
    using MmaSelector
        = detail::MmaSelector<Mma_impl, FragM, FragN, FragK, InputTA, InputTB, ComputeT>;

} // namespace rocwmma

#endif // ROCWMMA_MMA_SELECTOR_HPP
