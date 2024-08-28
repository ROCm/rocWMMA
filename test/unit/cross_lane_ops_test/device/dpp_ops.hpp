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

#ifndef ROCWMMA_DEVICE_DPP_OPS_HPP
#define ROCWMMA_DEVICE_DPP_OPS_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename>
    struct is_dpp_bcast : std::false_type
    {
    };

    template <uint32_t ElementIdx, uint32_t SubGroupSize, class BCastCtrl>
    struct is_dpp_bcast<DppImpl::OpsBase::BCast<ElementIdx, SubGroupSize, BCastCtrl>>
        : std::true_type
    {
    };

    template <typename>
    struct is_dpp_reverse : std::false_type
    {
    };

    template <uint32_t SubGroupSize, class ReverseCtrl>
    struct is_dpp_reverse<DppImpl::OpsBase::Reverse<SubGroupSize, ReverseCtrl>> : std::true_type
    {
    };

    template <typename>
    struct is_dpp_rotate : std::false_type
    {
    };

    template <uint32_t RotateDir, uint32_t RotateDist, uint32_t SubGroupSize, class RotateCtrl>
    struct is_dpp_rotate<DppImpl::OpsBase::Rotate<RotateDir, RotateDist, SubGroupSize, RotateCtrl>>
        : std::true_type
    {
    };

    template <typename>
    struct is_dpp_shift : std::false_type
    {
    };

    template <uint32_t ShiftDir, uint32_t ShiftDist, uint32_t SubGroupSize, class ShiftCtrl>
    struct is_dpp_shift<DppImpl::OpsBase::Shift<ShiftDir, ShiftDist, SubGroupSize, ShiftCtrl>>
        : std::true_type
    {
    };

    template <typename>
    struct is_dpp_shuffle2 : std::false_type
    {
    };

    template <uint32_t Select0, uint32_t Select1>
    struct is_dpp_shuffle2<DppImpl::OpsBase::Shuffle2<Select0, Select1>> : std::true_type
    {
    };

    template <typename>
    struct is_dpp_shuffle4 : std::false_type
    {
    };

    template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    struct is_dpp_shuffle4<DppImpl::OpsBase::Shuffle4<Select0, Select1, Select2, Select3>>
        : std::true_type
    {
    };

    template <typename>
    struct is_dpp_swap : std::false_type
    {
    };

    template <uint32_t SubGroupSize, class SwapCtrl>
    struct is_dpp_swap<DppImpl::OpsBase::Swap<SubGroupSize, SwapCtrl>> : std::true_type
    {
    };

    ROCWMMA_DEVICE inline bool isDppMasked(int id, uint32_t WriteRowMask, uint32_t WriteBankMask)
    {
        return (WriteRowMask & (1 << ((id >> 4) & 0x3)))
               && (WriteBankMask & (1 << ((id >> 2) & 0x3)));
    }

    template <int SubgroupSize, int Element>
    ROCWMMA_DEVICE inline int getDppBCastExpect(int input)
    {
        // TODO 63 should be waveSize - 1
        return (input & (~(SubgroupSize - 1))) + Element;
    }

    template <int SubgroupSize>
    ROCWMMA_DEVICE inline int getDppReverseExpect(int input)
    {
        int maxInGroup = SubgroupSize - 1;
        return ((input & (~maxInGroup) | (maxInGroup - (input & maxInGroup))));
    }

    template <int SubgroupSize, int Direction, int Distance>
    ROCWMMA_DEVICE inline int getDppRotateExpect(int input)
    {
        auto afterRotate = (input & (SubgroupSize - 1));
        afterRotate += Direction == CrossLaneOps::OP_DIR_L ? Distance : -Distance;
        afterRotate += SubgroupSize;
        afterRotate &= (SubgroupSize - 1);
        return (input & (~(SubgroupSize - 1))) | afterRotate;
    }

    template <uint32_t SubGroupSize, uint32_t ShiftDir, uint32_t ShiftDist, bool BoundCtrl>
    ROCWMMA_DEVICE inline int getDppShiftExpect(int input, int prev)
    {
        input += ShiftDir == CrossLaneOps::OP_DIR_L ? ShiftDist : -ShiftDist;
        int  fillValue  = BoundCtrl ? 0 : prev;
        auto afterShift = (input & (SubGroupSize - 1));
        afterShift += ShiftDir == CrossLaneOps::OP_DIR_L ? -ShiftDist : ShiftDist;
        return (afterShift < 0 || afterShift >= SubGroupSize) ? fillValue : input;
    }

    template <uint32_t Select0, uint32_t Select1>
    ROCWMMA_DEVICE inline int getDppShuffle2Expect(int input)
    {
        auto id = input & 0b1;
        input -= id;
        input += id == 0 ? Select0 : Select1;
        return input;
    }

    template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    ROCWMMA_DEVICE inline int getDppShuffle4Expect(int input)
    {
        auto id = input & 0b11;
        input -= id;
        input += id == 0 ? Select0 : id == 1 ? Select1 : id == 2 ? Select2 : Select3;
        return input;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline int getDppSwapExpect(int input)
    {
        return input ^ SubGroupSize;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<is_dpp_bcast<CrossLaneOp>::value, bool> dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect
            = isMasked ? getDppBCastExpect<CrossLaneOp::GROUP_SIZE, CrossLaneOp::ELEMENT_IDX>(input)
                       : prev;
        int output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op 0, input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n",  input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<is_dpp_reverse<CrossLaneOp>::value, bool> dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect   = isMasked ? getDppReverseExpect<CrossLaneOp::GROUP_SIZE>(input) : prev;
        int  output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op 0, input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n",  input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<is_dpp_rotate<CrossLaneOp>::value, bool> dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect   = isMasked ? getDppRotateExpect<CrossLaneOp::GROUP_SIZE,
                                                   CrossLaneOp::OP_DIR,
                                                   CrossLaneOp::OP_DIST>(input)
                                 : prev;
        int  output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op (%d, %d, %d), input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::OP_DIR, CrossLaneOp::OP_DIST, input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<is_dpp_shift<CrossLaneOp>::value, bool> dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect   = isMasked ? getDppShiftExpect<CrossLaneOp::GROUP_SIZE,
                                                  CrossLaneOp::OP_DIR,
                                                  CrossLaneOp::OP_DIST,
                                                  BoundCtrl>(input, prev)
                                 : prev;
        int  output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op (%d, %d, %d), input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n", CrossLaneOp::GROUP_SIZE, CrossLaneOp::OP_DIR, CrossLaneOp::OP_DIST, input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE
        std::enable_if_t<is_dpp_shuffle2<CrossLaneOp>::value || is_dpp_shuffle4<CrossLaneOp>::value,
                         bool>
        dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect   = -1;
        if constexpr(is_dpp_shuffle2<CrossLaneOp>::value)
        {
            expect = isMasked
                         ? getDppShuffle2Expect<CrossLaneOp::SELECT_0, CrossLaneOp::SELECT_1>(input)
                         : prev;
        }
        else if constexpr(is_dpp_shuffle4<CrossLaneOp>::value)
        {
            expect = isMasked ? getDppShuffle4Expect<CrossLaneOp::SELECT_0,
                                                     CrossLaneOp::SELECT_1,
                                                     CrossLaneOp::SELECT_2,
                                                     CrossLaneOp::SELECT_3>(input)
                              : prev;
        }
        int output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op (%d, %d), input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n", CrossLaneOp::SELECT_0, CrossLaneOp::SELECT_1, input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<is_dpp_swap<CrossLaneOp>::value, bool> dppOpsTestCase()
    {
        int  id       = threadIdx.x;
        int  prev     = 100; // TODO passed in by parameter
        int  input    = id;
        bool isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        int  expect   = isMasked ? getDppSwapExpect<CrossLaneOp::GROUP_SIZE>(input) : prev;
        int  output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(input,
                                                                                              prev);

        // printf("op (%d, %d), input %d, WriteRowMask %x, WriteBankMask %x, BoundCtrl %d, expect %d, output %d\n", CrossLaneOp::SELECT_0, CrossLaneOp::SELECT_1, input , WriteRowMask , WriteBankMask , BoundCtrl, expect , output );
        return output != expect;
    }
} // namespace rocwmma

#endif // ROCWMMA_DEVICE_DPP_OPS_HPP
