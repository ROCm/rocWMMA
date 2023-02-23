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
#ifndef ROCWMMA_PERMUTE_HPP
#define ROCWMMA_PERMUTE_HPP

#include "cross_lane_ops.hpp"
#include "permute_impl.hpp"

namespace rocwmma
{
    namespace PermuteOps
    {
        /**
         * \ingroup Cross-Lane Operations
         * \defgroup Permute Ops
         *
         * @brief Cross-lane operations implemented with the amdgcn_ds_permute and amdgcn_ds_bpermute backends.
         *
         * Here we build out the cross-lane properties specific to permute, such as the backend (OP_IMPL_PERMUTE).
         *
         * These definitions are for permute support, so we specify this backend here. These ops must
         * inherit the meta-data front end from CrossLaneOps, AND the thread index calculation from the backend.
         */

        // clang-format off

        using CrossLaneOps::Properties;
        using CrossLaneOps::BlockBCast;
        using CrossLaneOps::OpBase;
        using CrossLaneOps::RotateL;
        using CrossLaneOps::RotateR;

        constexpr uint32_t OP_IMPL_PERM  = Properties::OP_IMPL_PERMUTE;
        constexpr uint32_t OP_IMPL_BPERM  = Properties::OP_IMPL_BPERMUTE;
        constexpr uint32_t OP_CTRL = 0x0; // Uses thread index calculation instead

        template<uint32_t BlockIdx>
        struct BlockBCast32 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_32, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_32, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast16 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_16, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_16, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast8 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_8, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_8, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast4 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_4, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_4, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast2 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_2, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_2, BlockIdx>{};


        template<uint32_t VW, uint32_t ElementShift>
        struct Gather32 : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_32, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_interleave<Properties::OP_GROUP_SIZE_32, VW, ElementShift>{};

        template<uint32_t VW, uint32_t ElementShift>
        struct Scatter32 : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_32, OP_IMPL_PERM, OP_CTRL>, detail::amdgcn_interleave<Properties::OP_GROUP_SIZE_32, VW, ElementShift>{};

        template<uint32_t VW, uint32_t ElementShift>
        struct Gather16 : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_16, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_interleave<Properties::OP_GROUP_SIZE_16, VW, ElementShift>{};

        template<uint32_t VW, uint32_t ElementShift>
        struct Scatter16 : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_16, OP_IMPL_PERM, OP_CTRL>, detail::amdgcn_interleave<Properties::OP_GROUP_SIZE_16, VW, ElementShift>{};


        template<uint32_t Distance>
        struct RotateWaveL : RotateL<Distance, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_rotate<Properties::OP_GROUP_SIZE_WARP, Distance>{};
        template<uint32_t Distance>
        struct RotateWaveR : RotateR<Distance, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_PERM, OP_CTRL>, detail::amdgcn_rotate<Properties::OP_GROUP_SIZE_WARP, Distance>{};


        template<uint32_t BlockSize>
        struct DupLoBlockWave : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<Properties::OP_GROUP_SIZE_WARP, BlockSize, 2u, 0u>{};

        template<uint32_t BlockSize>
        struct DupHiBlockWave : OpBase<Properties::OP_ID_SHUFFLE, Properties::OP_GROUP_SIZE_WARP, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<Properties::OP_GROUP_SIZE_WARP, BlockSize, 2u, Properties::OP_GROUP_SIZE_WARP / BlockSize / 2u>{};

        template<uint32_t GroupSize, uint32_t BlockSize, uint32_t DupCount, uint32_t Shift>
        struct DupTest : OpBase<Properties::OP_ID_SHUFFLE, GroupSize, OP_IMPL_BPERM, OP_CTRL>, detail::amdgcn_duplicate_blocks<GroupSize, BlockSize, DupCount, Shift>{};

        // clang-format on
    }

    template <typename PermuteOp>
    struct Permute
    {
        using PermuteFunc =
            typename std::conditional_t<PermuteOp::opImpl()
                                            == CrossLaneOps::Properties::OP_IMPL_PERMUTE,
                                        detail::amdgcn_ds_permute,
                                        detail::amdgcn_ds_bpermute>;

        // Sanity checks
        static_assert((PermuteOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_PERMUTE)
                          || (PermuteOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_BPERMUTE),
                      "PermuteOp must use permute or permute backend");
        static_assert((PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_BLOCK_BCAST)
                          || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE),
                      "PermuteOp is unsupported");

        template <typename DataT>
        ROCWMMA_DEVICE static inline auto exec(DataT const& src)
        {
            return PermuteFunc::exec(src,
                                     PermuteOp::threadCtrl(detail::WaveSpace<>::localLaneId()));
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize>& src)
        {
            auto it = makeVectorIterator(src).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = PermuteFunc::exec(*it,
                                        PermuteOp::threadCtrl(detail::WaveSpace<>::localLaneId()));
                it++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_HPP
