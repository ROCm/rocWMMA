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
#ifndef ROCWMMA_BLEND_HPP
#define ROCWMMA_BLEND_HPP

#include "blend_impl.hpp"
#include "cross_lane_ops.hpp"
namespace rocwmma
{
    template <typename BlendOp>
    struct Blend
    {

    public:
        // Sanity checks
        static_assert((BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VPERM)
                          || (BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VBLEND),
                      "BlendOp must use vperm or blend backend");
        static_assert((BlendOp::opId() == CrossLaneOps::Properties::OP_ID_BLEND)
                          || (BlendOp::opId() == CrossLaneOps::Properties::OP_ID_PERM_BYTE),
                      "BlendOp is unsupported");

        template <typename DataT>
        ROCWMMA_DEVICE static inline DataT exec(DataT const& src0, DataT const& src1)
        {
            return BlendOp::exec(src0, src1);
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>& src0, DataT const& src1)
        {
            auto it = makeVectorIterator(src0).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = exec(*it, src1);
                it++;
            }
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>&       src0,
                                        VecT<DataT, VecSize> const& src1)
        {
            auto       it0 = makeVectorIterator(src0).begin();
            auto const it1 = makeVectorIterator(src1).begin();
            static_assert(decltype(it0)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it0 = exec(*it0, *it1);
                it0++;
                it1++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_BLEND_HPP
