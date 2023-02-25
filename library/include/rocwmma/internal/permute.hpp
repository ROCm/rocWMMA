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
    template <typename PermuteOp>
    struct Permute
    {
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
            return PermuteOp::exec(src, detail::WaveSpace<>::localLaneId());
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
                *it = exec(*it);
                it++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_HPP
