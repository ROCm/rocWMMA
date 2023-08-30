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
#ifndef ROCWMMA_PERMUTE_IMPL_HPP
#define ROCWMMA_PERMUTE_IMPL_HPP

#include "permute.hpp"

namespace rocwmma
{

    namespace PermuteImpl
    {
        // Implementation meta-data
        using CrossLaneOps::OpBase;
        using CrossLaneOps::Properties;

        // Dpp backend
        using Properties::OP_IMPL_BPERMUTE;
        using Properties::OP_IMPL_PERMUTE;

        // Functional
        using Properties::OP_ID_BLOCK_BCAST;
        using Properties::OP_ID_GATHER;
        using Properties::OP_ID_ROTATE;
        using Properties::OP_ID_SCATTER;

        // Groups
        using Properties::OP_GROUP_SIZE_16;
        using Properties::OP_GROUP_SIZE_2;
        using Properties::OP_GROUP_SIZE_32;
        using Properties::OP_GROUP_SIZE_4;
        using Properties::OP_GROUP_SIZE_8;
        using Properties::OP_GROUP_SIZE_WARP;

        // Detail
        using Properties::OP_DIR_L;
        using Properties::OP_DIR_R;

        namespace Backend
        {
            // bpermute: for the current thread, read from laneId
            template <class BPermuteCtrl>
            struct amdgcn_ds_bpermute
            {
                template <typename InputT>
                ROCWMMA_DEVICE static inline InputT exec(InputT input, uint32_t laneId)
                {
                    // NOTE: final address is laneId * 4
                    reinterpret_cast<uint32_t&>(input)
                        = __builtin_amdgcn_ds_bpermute((BPermuteCtrl::threadCtrl(laneId) << 2),
                                                       reinterpret_cast<uint32_t const&>(input));
                    return input;
                }
            };

            // permute: for the current thread, push my value to laneId
            template <class PermuteCtrl>
            struct amdgcn_ds_permute
            {
                template <typename InputT>
                ROCWMMA_DEVICE static inline InputT exec(InputT input, uint32_t laneId)
                {
                    // NOTE: final address is laneId * 4
                    reinterpret_cast<uint32_t&>(input)
                        = __builtin_amdgcn_ds_permute((PermuteCtrl::threadCtrl(laneId) << 2),
                                                      reinterpret_cast<uint32_t const&>(input));
                    return input;
                }
            };

        } // namespace Backend

        namespace Ctrl
        {
            template <uint32_t BlockIdx, uint32_t BlockSize>
            struct BPermuteBlockBCast
            {
            private:
                enum Traits : uint32_t
                {
                    BLOCK_OFFSET = BlockIdx * BlockSize,
                };

            public:
                // Calculate the read element based on thread position.
                ROCWMMA_DEVICE static inline uint32_t threadCtrl(uint32_t threadId)
                {
                    // Make sure that the threadId is within range
                    auto tIdx = threadId % BlockSize;
                    return Traits::BLOCK_OFFSET + tIdx;
                }
            };

            template <uint32_t BlockSize, uint32_t VW, uint32_t ElementShift>
            struct Interleave
            {
            private:
                enum Traits : uint32_t
                {
                    MASK_0 = LsbMask<Log2<BlockSize>::value>::value,
                    MASK_1 = LsbMask<Log2<BlockSize>::value - Log2<VW>::value>::value,
                    MASK_2 = ~MASK_0
                };

            public:
                // Calculate the read element based on thread position.
                ROCWMMA_HOST_DEVICE static inline uint32_t threadCtrl(uint32_t threadId)
                {
                    // For reference in readibility:
                    // const uint32_t offset0 = (threadId * BlockSize / VW + ElementShift) % BlockSize;
                    // const uint32_t offset1 = threadId / VW % (BlockSize/VW);
                    // const uint32_t offset2 = (threadId / BlockSize) * BlockSize;

                    const uint32_t offset0
                        = ((threadId << (Log2<BlockSize>::value - Log2<VW>::value)) + ElementShift)
                          & Traits::MASK_0;
                    const uint32_t offset1 = (threadId >> Log2<VW>::value) & Traits::MASK_1;
                    const uint32_t offset2 = threadId & Traits::MASK_2;
                    return offset0 + offset1 + offset2;
                }
            };

            template <uint32_t GroupSize, uint32_t Distance>
            struct Rotate
            {
            private:
                enum Traits : uint32_t
                {
                    MASK_0 = LsbMask<Log2<GroupSize>::value>::value,
                    MASK_1 = ~MASK_0
                };

            public:
                // Calculate the read element based on thread position.
                ROCWMMA_HOST_DEVICE static inline uint32_t threadCtrl(uint32_t threadId)
                {
                    // For reference in readibility:
                    // const uint32_t offset0 = (threadId + Distance) % GroupSize
                    // const uint32_t offset1 = (threadId / GroupSize) * GroupSize
                    const uint32_t offset0 = (threadId + Distance) & Traits::MASK_0;
                    const uint32_t offset1 = threadId & Traits::MASK_1;
                    return offset0 + offset1;
                }
            };

            template <uint32_t GroupSize, uint32_t BlockSize, uint32_t DupCount, uint32_t Offset>
            struct DuplicateBlocks
            {
            private:
                enum Traits : uint32_t
                {
                    MASK_0 = LsbMask<Log2<BlockSize>::value>::value,
                    MASK_1 = LsbMask<Log2<GroupSize>::value - Log2<DupCount>::value>::value,
                    MASK_2 = ~LsbMask<Log2<GroupSize>::value>::value,
                };

            public:
                // Calculate the read element based on thread position.
                ROCWMMA_HOST_DEVICE static inline uint32_t threadCtrl(uint32_t threadId)
                {
                    // For reference in readibility:
                    // const uint32_t offset0 = (threadId % BlockSize)
                    // const uint32_t offset1 = ((threadId / DupCount / BlockSize ) * BlockSize) % (GroupSize / DupCount)
                    // const uinti32_t offset2 = (threadId / GroupSize) * GroupSize
                    // constexpr uint32_t offset3 = Offset * BlockSize
                    const uint32_t offset0 = (threadId & Traits::MASK_0);
                    const uint32_t offset1
                        = (threadId >> Log2<DupCount>::value) & ~(~Traits::MASK_0 ^ Traits::MASK_1);
                    const uint32_t     offset2 = (threadId & Traits::MASK_2);
                    constexpr uint32_t offset3 = Offset << Log2<BlockSize>::value;
                    return offset0 + offset1 + offset2 + offset3;
                }
            };

        } // namespace Ctrl
        namespace OpsBase
        {
            template <uint32_t OpId, uint32_t SubGroupSize>
            using PermuteOp = OpBase<OpId, SubGroupSize, OP_IMPL_PERMUTE>;

            template <uint32_t OpId, uint32_t SubGroupSize>
            using BPermuteOp = OpBase<OpId, SubGroupSize, OP_IMPL_BPERMUTE>;

            template <uint32_t BlockIdx, uint32_t BlockSize>
            struct BlockBCast
                : public BPermuteOp<OP_ID_BLOCK_BCAST, BlockSize>,
                  public Backend::amdgcn_ds_bpermute<Ctrl::BPermuteBlockBCast<BlockIdx, BlockSize>>
            {
                enum : uint32_t
                {
                    ELEMENT_IDX = BlockIdx,
                };

                constexpr static uint32_t elementIdx()
                {
                    return ELEMENT_IDX;
                }
            };

            template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
            struct Gather
                : public BPermuteOp<OP_ID_GATHER, SubGroupSize>,
                  public Backend::amdgcn_ds_bpermute<Ctrl::Interleave<SubGroupSize, VW, Shift>>
            {
            };

            template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
            struct Scatter
                : public PermuteOp<OP_ID_SCATTER, SubGroupSize>,
                  public Backend::amdgcn_ds_permute<Ctrl::Interleave<SubGroupSize, VW, Shift>>
            {
            };

            /*! \class Rotate
            *  \brief Perform element-wise rotation in direction \p RotateDir in sub-groups of \p SubGroupSize threads.
            *
            * @tparam RotateDir rotation direction: see Properties
            * @tparam RotateDistance element positions to move in specified direction. Positions wrapped by sub group size.
            */
            template <uint32_t RotateDist, uint32_t SubGroupSize>
            struct RotateL
                : public BPermuteOp<OP_ID_ROTATE, SubGroupSize>,
                  public Backend::amdgcn_ds_bpermute<Ctrl::Rotate<SubGroupSize, RotateDist>>
            {
                enum : uint32_t
                {
                    OP_DIR  = OP_DIR_L,
                    OP_DIST = RotateDist
                };

                constexpr static uint32_t opDir()
                {
                    return OP_DIR;
                }
                constexpr static uint32_t opDist()
                {
                    return OP_DIST;
                }
            };

            template <uint32_t RotateDist, uint32_t SubGroupSize>
            struct RotateR
                : public PermuteOp<OP_ID_ROTATE, SubGroupSize>,
                  public Backend::amdgcn_ds_permute<Ctrl::Rotate<SubGroupSize, RotateDist>>
            {
                enum : uint32_t
                {
                    OP_DIR  = OP_DIR_R,
                    OP_DIST = RotateDist
                };

                constexpr static uint32_t opDir()
                {
                    return OP_DIR;
                }
                constexpr static uint32_t opDist()
                {
                    return OP_DIST;
                }
            };
        }

        namespace Ops
        {
            /**
             * \ingroup Cross_Lane_Operations
             *
             * @brief Cross-lane operations implemented with the amdgcn_ds_permute and amdgcn_ds_bpermute backends.
             *
             * Here we build out the cross-lane properties specific to permute, such as the backend (OP_IMPL_PERMUTE).
             *
             * These definitions are for permute support, so we specify this backend here. These ops must
             * inherit the meta-data front end from CrossLaneOps, AND the thread index calculation from the backend.
             */

            // clang-format off

            template<uint32_t BlockIdx>
            using BlockBCast32 = OpsBase::BlockBCast<BlockIdx, OP_GROUP_SIZE_32>;

            template<uint32_t BlockIdx>
            using BlockBCast16 = OpsBase::BlockBCast<BlockIdx, OP_GROUP_SIZE_16>;

            template<uint32_t BlockIdx>
            using BlockBCast8 = OpsBase::BlockBCast<BlockIdx, OP_GROUP_SIZE_8>;

            template<uint32_t BlockIdx>
            using BlockBCast4 = OpsBase::BlockBCast<BlockIdx, OP_GROUP_SIZE_4>;

            template<uint32_t BlockIdx>
            using BlockBCast2 = OpsBase::BlockBCast<BlockIdx, OP_GROUP_SIZE_2>;


            template<uint32_t VW, uint32_t ElementShift>
            using GatherWave = OpsBase::Gather<OP_GROUP_SIZE_WARP, VW, ElementShift>;

            template<uint32_t VW, uint32_t ElementShift>
            using Gather32 = OpsBase::Gather<OP_GROUP_SIZE_32, VW, ElementShift>;

            template<uint32_t VW, uint32_t ElementShift>
            using Gather16 = OpsBase::Gather<OP_GROUP_SIZE_16, VW, ElementShift>;

            template<uint32_t VW, uint32_t ElementShift>
            using ScatterWave = OpsBase::Gather<OP_GROUP_SIZE_WARP, VW, ElementShift>;

            template<uint32_t VW, uint32_t ElementShift>
            using Scatter32 = OpsBase::Gather<OP_GROUP_SIZE_32, VW, ElementShift>;

            template<uint32_t VW, uint32_t ElementShift>
            using Scatter16 = OpsBase::Gather<OP_GROUP_SIZE_16, VW, ElementShift>;


            template<uint32_t Distance>
            using RotateWaveL = OpsBase::RotateL<Distance, OP_GROUP_SIZE_WARP>;

            template<uint32_t Distance>
            using RotateWaveR = OpsBase::RotateR<Distance, OP_GROUP_SIZE_WARP>;


            // template<uint32_t BlockSize>
            // struct DupLoBlockWave : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<Properties::OP_GROUP_SIZE_WARP, BlockSize, 2u, 0u>{};

            // template<uint32_t BlockSize>
            // struct DupHiBlockWave : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<Properties::OP_GROUP_SIZE_WARP, BlockSize, 2u, Properties::OP_GROUP_SIZE_WARP / BlockSize / 2u>{};

            // template<uint32_t GroupSize, uint32_t BlockSize, uint32_t DupCount, uint32_t Shift>
            // struct DupTest : OpBase<Properties::OP_ID_SHUFFLE, GroupSize, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<GroupSize, BlockSize, DupCount, Shift>{};

            // clang-format on
        } // namespace Ops

    } // namespace PermuteImpl

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_IMPL_HPP
