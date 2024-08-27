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
    ROCWMMA_DEVICE inline bool isDppMasked(int id, uint32_t WriteRowMask, uint32_t WriteBankMask)
    {
        return (WriteRowMask & (1 << ((id >> 4) & 0x3)))
               && (WriteBankMask & (1 << ((id >> 2) & 0x3)));
    }

    template <int SubgroupSize, int Element>
    ROCWMMA_DEVICE inline int getDppBCastExpect(int input)
    {
        // TODO 63 should be waveSize - 1
        return (input & (63 & (~(SubgroupSize - 1)))) + Element;
    }

    template <int SubgroupSize>
    ROCWMMA_DEVICE inline int getDppReverseExpect(int input)
    {
        int maxInGroup = SubgroupSize - 1;
        return ((input & (~maxInGroup) | (maxInGroup - (input & maxInGroup))));
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast2<0>>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast2<1>>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast4<0>>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast4<3>>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast16<2>>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::BCast16<11>>,
                                    bool>
                   dppOpsTestCase()
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
    ROCWMMA_DEVICE std::enable_if_t<std::is_same_v<CrossLaneOp, DppImpl::Ops::Reverse2>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::Reverse4>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::Reverse8>
                                        || std::is_same_v<CrossLaneOp, DppImpl::Ops::Reverse16>,
                                    bool>
                   dppOpsTestCase()
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
} // namespace rocwmma

#endif // ROCWMMA_DEVICE_DPP_OPS_HPP
