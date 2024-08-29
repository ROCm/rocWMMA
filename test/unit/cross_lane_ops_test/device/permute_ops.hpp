/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_DEVICE_PERMUTE_OPS_HPP
#define ROCWMMA_DEVICE_PERMUTE_OPS_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename>
    struct is_permute_blockBCast : std::false_type
    {
    };

    template <uint32_t BlockIdx, uint32_t BlockSize>
    struct is_permute_blockBCast<PermuteImpl::OpsBase::BlockBCast<BlockIdx, BlockSize>>
        : std::true_type
    {
    };

    template <uint32_t BlockIdx, uint32_t BlockSize>
    ROCWMMA_DEVICE inline int getPermuteBlockBCastExpect(int input)
    {
        auto idxInBlock = input % BlockSize;
        input           = BlockIdx * BlockSize + idxInBlock;
        return input;
    }

    template <int SubgroupSize, int Direction, int Distance>
    ROCWMMA_DEVICE inline int getPermuteRotateExpect(int input)
    {
        auto afterRotate = (input & (SubgroupSize - 1));
        afterRotate += Direction == CrossLaneOps::OP_DIR_L ? Distance : -Distance;
        afterRotate += SubgroupSize;
        afterRotate &= (SubgroupSize - 1);
        return (input & (~(SubgroupSize - 1))) | afterRotate;
    }

    template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
    ROCWMMA_DEVICE inline int getPermuteGatherExpect(int input)
    {
        auto idxInGroup = input % SubGroupSize;
        input -= idxInGroup;
        return input
               + (((idxInGroup % VW) * (SubGroupSize / VW) + idxInGroup / VW + Shift)
                  % SubGroupSize);
    }

    template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
    ROCWMMA_DEVICE inline int getPermuteScatterExpect(int input)
    {
        return 0;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<is_permute_blockBCast<CrossLaneOp>::value, bool>
                   permuteOpsTestCase()
    {
        int input = threadIdx.x;
        int expect
            = getPermuteBlockBCastExpect<CrossLaneOp::ELEMENT_IDX, CrossLaneOp::GROUP_SIZE>(input);
        int output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        // printf("op (%d, %d), input %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::ELEMENT_IDX, input , expect , output );
        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_BPERMUTE,
                                    bool>
                   permuteOpsTestCase()
    {
        int input  = threadIdx.x;
        int expect = getPermuteRotateExpect<CrossLaneOp::GROUP_SIZE,
                                            CrossLaneOps::OP_DIR_L,
                                            CrossLaneOp::opDist()>(input);
        int output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        // printf("op (%d, %d, %d), input %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::opDir(), CrossLaneOp::opDist(), input , expect , output );
        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_PERMUTE,
                                    bool>
                   permuteOpsTestCase()
    {
        int input  = threadIdx.x;
        int expect = getPermuteRotateExpect<CrossLaneOp::GROUP_SIZE,
                                            CrossLaneOps::OP_DIR_R,
                                            CrossLaneOp::opDist()>(input);
        int output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        // printf("op (%d, %d, %d), input %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::opDir(), CrossLaneOp::opDist(), input , expect , output );
        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_GATHER
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_BPERMUTE,
                                    bool>
                   permuteOpsTestCase()
    {
        int input  = threadIdx.x;
        int expect = getPermuteGatherExpect<CrossLaneOp::GROUP_SIZE,
                                            CrossLaneOp::vw(),
                                            CrossLaneOp::shift()>(input);
        int output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        // printf("op (%d, %d, %d), input %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::vw(), CrossLaneOp::shift(), input , expect , output );
        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SCATTER
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_PERMUTE,
                                    bool>
                   permuteOpsTestCase()
    {
        int input  = threadIdx.x;
        int expect = getPermuteScatterExpect<CrossLaneOp::GROUP_SIZE,
                                             CrossLaneOp::vw(),
                                             CrossLaneOp::shift()>(input);
        int output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        // printf("op (%d, %d, %d), input %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::opDir(), CrossLaneOp::opDist(), input , expect , output );
        return output != expect;
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_PERMUTE_OPS_HPP
