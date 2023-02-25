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
#ifndef ROCWMMA_SWIZZLE_HPP
#define ROCWMMA_SWIZZLE_HPP

#include "cross_lane_ops.hpp"
#include "swizzle_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{

    /**
     * \ingroup Swizzle Ops
     * \defgroup Swizzle Front-End
     *
     * @brief Cross-lane operations implemented with the amdgcn_ds_swizzle backend.
     *
     * This function does not use LDS memory, but does use the LDS hardware, therefore it will
     * implicitly require lgkmcnt waits. This function is significantly faster than reading or
     * writing to the LDS memory, but does introduce lgkmcnt dependence for data cohesion.
     * This is not the most performant backend (dpp can be faster), however there is better
     * flexibility with swizzle when necessary. Fft swizzle is also unique to this backend.
     *
     * The swizzle backend offers a set of cross-lane support:
     * [X, Y] = Includes X and Y
     * [X - Y] = X, powers of 2 in between and including Y
     *
     * BCast (Subgroups[2 - 32])
     * Reverse (Subgroups[2 - 32])
     * Rotate (L/R, Subgroups[2 - 32])
     * Shuffle (Subgroups[2, 4])
     * Swap (Subgroups[2 - 16])
     *
     * Fft (FftCtrl [0x00 - 0x1F]) -> See ISA for swizzle fft codes
     *
     * The swizzle backend does not support Shift.
     */

    /*! \class Swizzle
    *  \brief A front-end utility that invokes swizzle operations on input data.
    *
    * @tparam SwizzleOp - fully qualified op class that generates the OP_CTRL code to apply to Dpp function.
    */
    template <typename SwizzleOp>
    struct Swizzle
    {
        // Sanity checks
        static_assert(SwizzleOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_SWIZZLE,
                      "SwizzleOp must use swizzle backend");
        static_assert((SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                          || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                          || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_REVERSE)
                          || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_SWAP)
                          || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                          || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_FFT),
                      "SwizzleOp is unsupported");

        template <typename DataT>
        ROCWMMA_DEVICE static auto exec(DataT const& src)
        {
            return SwizzleOp::exec(src);
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>& src)
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

#endif // ROCWMMA_SWIZZLE_HPP
