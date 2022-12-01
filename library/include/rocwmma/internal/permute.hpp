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

        constexpr uint32_t OP_IMPL  = Properties::OP_IMPL_PERMUTE;
        constexpr uint32_t OP_CTRL = 0x0; // Uses thread index calculation instead

        template<uint32_t BlockIdx>
        struct BlockBCast32 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_32, OP_IMPL, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_32, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast16 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_16, OP_IMPL, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_16, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast8 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_8, OP_IMPL, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_8, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast4 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_4, OP_IMPL, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_4, BlockIdx>{};

        template<uint32_t BlockIdx>
        struct BlockBCast2 : BlockBCast<BlockIdx, Properties::OP_GROUP_SIZE_2, OP_IMPL, OP_CTRL>, detail::amdgcn_bpermute_block_bcast<Properties::OP_GROUP_SIZE_2, BlockIdx>{};

        // clang-format on
    }

    template <typename PermuteOp>
    struct Permute
    {
        using PermuteFunc = detail::amdgcn_ds_bpermute;

        // Sanity checks
        static_assert(PermuteOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_PERMUTE,
                      "PermuteOp must use permute backend");
        static_assert(PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_BLOCK_BCAST,
                      "PermuteOp is unsupported");

        template <typename DataT>
        __host__ __device__ static DataT exec(DataT const& src, uint32_t laneId)
        {
            return PermuteFunc::exec(src, PermuteOp::threadCtrl(laneId));
        }

        template <typename DataT>
        __host__ __device__ static void exec(DataT& src, uint32_t laneId)
        {
            src = PermuteFunc::exec(src, PermuteOp::threadCtrl(laneId));
        }

        template <typename DataT, uint32_t VecSize>
        __host__ __device__ static void exec(VecT<DataT, VecSize>& src, uint32_t laneId)
        {
            auto it = makeVectorIterator(src).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = PermuteFunc::exec(*it, PermuteOp::threadCtrl(laneId));
                it++;
            }
        }

        template <typename DataT, uint32_t VecSize>
        __host__ __device__ static auto exec(VecT<DataT, VecSize> const& src, uint32_t laneId)
        {
            VecT<DataT, VecSize> result;
            auto                 itW = makeVectorIterator(result).begin();
            auto const           itR = makeVectorIterator(src).begin();
            static_assert(decltype(itW)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *itW = PermuteFunc::exec(*itR, PermuteOp::threadCtrl(laneId));
                itW++;
                itR++;
            }
            return result;
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_HPP
