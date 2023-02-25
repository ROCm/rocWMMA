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
#ifndef ROCWMMA_DPP_HPP
#define ROCWMMA_DPP_HPP

#include "cross_lane_ops.hpp"
#include "dpp_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{
    /**
     * \ingroup Dpp Ops
     * \defgroup Dpp Front-End
     *
     * @brief Cross-lane operations implemented with the amdgcn_mov_dpp backend.
     *
     * This function does not use LDS memory or LDS hardware, therefore does not
     * implicitly require lgkmcnt waits. This is the fastest cross-lane function,
     * however is not as flexible as others.
     *
     * The dpp backend offers a set of cross-lane support:
     * [X, Y] = Includes X and Y
     * [X - Y] = X, powers of 2 in between and including Y
     *
     * BCast (Subgroups[2, 4, 16])
     * BCast16x15 -> waterfall broadcast (see DppOps)
     * BCast32x31 -> waterfall broadcast (see DppOps)
     *
     * Reverse (Subgroups[2-16])
     *
     * RotateR (Subgroups[2, 4, 16, WaveSize*])
     * RotateL (Subgroups[2, 4, WaveSize*])
     *
     * ShiftR (Subgroups[16], WaveSize*)
     * ShiftL (Subgroups[16], WaveSize*)
     *
     * Shuffle (Subgroups[2, 4])
     *
     * Swap (Subgroups[2])
     *
     * WaveSize* = architecture wave size (wave64 for gfx9 and wave32 for gfx11)
     *
     * The dpp backend does not support Fft. It also has limited group size support for
     * functionalities.
     *
     *
     * @note In this context:
     * 'row' means sub-group size of 16 elements. Wave64 has 4 rows, Wave32 has 2 rows per register.
     * 'bank' means sub-group size of 4 elements. There are 4 banks per row.
     * DPP (Data Parallel Primitives) has the added capacity of masking outputs of the function.
     *
     * WriteRowMask (4-bits), represents write control to each sub-group of 16 elements in a 64-wide register.
     * E.g. 0xF = write to all rows, 0x3 = write to only first two rows in each register.
     * Rows not written to will carry forward the 'prev' input value.
     *
     * WriteBankMask (4-bits), represents write control to each sub-group of 4 elements in a 16-wide row.
     * E.g. 0xF = write to all banks, 0x3 = write to only first two banks in each row.
     * Banks not written to will carry forward the 'prev' input value.
     *
     * BoundCtrl (1 bit), represents whether out-of-bounds indices will overwrite the output with 0.
     * E.g. Executing a shuffle4 with Select0 = 10 will result in 0 as the first element
     * in each active bank in each active row written to output.
     *
     * 'prev' value passed in may be a scalar, another vector, or the same as 'input'
     */

    /*! \class Dpp
    *  \brief A front-end utility that invokes Dpp (Data-parallel primitives) operations on input data.
    * Unique to Dpp is ability to write-mask resulting outputs of every op.
    *
    * @tparam DppOp - fully qualified op class that generates the OP_CTRL code to apply to Dpp function.
    * @tparam WriteRowMask - Mask output write rows (0 - disable write, 1 - enable write) wave64[3:0] wave32[1:0]
    * @tparam WriteBankMask - Mask output write banks (0 - disable write, 1 - enable write) wave64 and wave32[3:0]
    * @tparam BoundCtrl - OOB thread indices write 0 to output element
    */
    template <typename DppOp,
              uint32_t WriteRowMask  = 0xF,
              uint32_t WriteBankMask = 0xF,
              bool     BoundCtrl     = false>
    struct Dpp
    {
        // Sanity checks
        static_assert(DppOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP,
                      "DppOp must use dpp backend");
        static_assert((DppOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SHIFT)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_REVERSE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SWAP)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_WFALL_BCAST)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_MOVE),
                      "DppOp is unsupported");

        template <typename DataT>
        ROCWMMA_DEVICE static inline DataT exec(DataT const& src, DataT const& prev)
        {
            return DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(src, prev);
        }

        template <typename DataT>
        ROCWMMA_DEVICE static inline DataT exec(DataT const& src)
        {
            return DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(src, src);
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline void exec(VecT<DataT, VecSize>& src)
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

        // Scalar as prev
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline void exec(VecT<DataT, VecSize>& src, DataT const& prev)
        {
            auto it = makeVectorIterator(src).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = exec(*it, prev);
                it++;
            }
        }

        // Vector as prev
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline void exec(VecT<DataT, VecSize>&       src,
                                               VecT<DataT, VecSize> const& prev)
        {
            auto       it  = makeVectorIterator(src).begin();
            const auto itp = makeVectorIterator(prev).begin();

            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = exec(*it, *itp);
                it++;
                itp++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DPP_HPP
