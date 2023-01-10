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

    namespace DppOps
    {
        /**
         * \ingroup Cross-Lane Operations
         * \defgroup Dpp Ops
         *
         * @brief Cross-lane operations implemented with the amdgcn_mov_dpp backend.
         * @note In this context:
         * 'row' means sub-group size of 16 elements. Wave64 has 4 rows, Wave32 has 2 rows per register.
         * 'bank' means sub-group size of 4 elements. There are 4 banks per row.
         *
         * DPP (Data Parallel Primitives) backend can implement specific variations of the cross-lane operations
         * so we can fully specialize those here.
         *
         * Here we build out the cross-lane properties specific to DPP, such as the backend (OP_IMPL) and the
         * control code drivers for the backend function call (OP_CTRL). Control code generators are implemented
         * in the DppCtrl namespace.
         */

        // clang-format off

        using CrossLaneOps::Properties;
        using CrossLaneOps::BCast;
        using CrossLaneOps::WFallBCast;
        using CrossLaneOps::Reverse;
        using CrossLaneOps::RotateR;
        using CrossLaneOps::RotateL;
        using CrossLaneOps::ShiftR;
        using CrossLaneOps::ShiftL;
        using CrossLaneOps::Swap;

        // These definitions are for DPP support, so we specify this backend here.
        // Operation directions can also be propagated to the ctrl
        constexpr uint32_t OP_IMPL  = Properties::OP_IMPL_DPP;
        constexpr uint32_t OP_DIR_L = Properties::OP_DIR_L;
        constexpr uint32_t OP_DIR_R = Properties::OP_DIR_R;

        // BCast variants
        template <uint32_t ElementIdx>
        using BCast16 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_bcast<ElementIdx>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast4 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::DppCtrl::amdgcn_dpp_shuffle_4<ElementIdx,
                                                                                                                    ElementIdx,
                                                                                                                    ElementIdx,
                                                                                                                    ElementIdx>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast2 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::DppCtrl::amdgcn_dpp_shuffle_4<ElementIdx,
                                                                                                                    ElementIdx,
                                                                                                                    ElementIdx + Properties::OP_GROUP_SIZE_2,
                                                                                                                    ElementIdx + Properties::OP_GROUP_SIZE_2>::opCtrl()>;

        // Special BCast variants:
        // BCast<M>x<N>, where:
        // <M> = subgroup size
        // <N> = element idx
        // NOTE: These functions only broadcast the <N>th element of the current subgroup to the NEXT subgroup
        using BCast16x15 = WFallBCast<Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_bcast15::opCtrl()>;

        using BCast32x31 = WFallBCast<Properties::OP_GROUP_SIZE_32, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_bcast31::opCtrl()>;


        // Reversal variants
        using Reverse16 = Reverse<Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_reverse::opCtrl()>;

        using Reverse8 = Reverse<Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::DppCtrl::amdgcn_dpp_half_row_reverse::opCtrl()>;

        using Reverse4 = Reverse<Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::DppCtrl::amdgcn_dpp_shuffle_4<0x3, 0x2, 0x1, 0x0>::opCtrl()>;

        using Reverse2 = Reverse<Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::DppCtrl::amdgcn_dpp_shuffle_4<0x1, 0x0, 0x3, 0x2>::opCtrl()>;


        /// Rotation variants

        // Rotate the entire wave by 1
        using RotateWaveR1 = RotateR<1u, Properties::OP_GROUP_SIZE_WARP, OP_IMPL, detail::DppCtrl::amdgcn_dpp_wave_rotate1<OP_DIR_R>::opCtrl()>;

        using RotateWaveL1 = RotateL<1u, Properties::OP_GROUP_SIZE_WARP, OP_IMPL, detail::DppCtrl::amdgcn_dpp_wave_rotate1<OP_DIR_L>::opCtrl()>;

        // Rotate in element groups
        template <uint32_t RotateDistance>
        using RotateR16 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_rotate_r<RotateDistance>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL4 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::DppCtrl::amdgcn_dpp_bank_rotate<OP_DIR_L, RotateDistance>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR4 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::DppCtrl::amdgcn_dpp_bank_rotate<OP_DIR_R, RotateDistance>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL2 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::DppCtrl::amdgcn_dpp_half_bank_rotate<OP_DIR_L, RotateDistance>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR2 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::DppCtrl::amdgcn_dpp_half_bank_rotate<OP_DIR_R, RotateDistance>::opCtrl()>;

        /// Shift variants

        // Rotate the entire wave by 1
        using ShiftWaveL1 = ShiftL<1u, Properties::OP_GROUP_SIZE_WARP, OP_IMPL, detail::DppCtrl::amdgcn_dpp_wave_shift1<OP_DIR_L>::opCtrl()>;

        using ShiftWaveR1 = ShiftR<1u, Properties::OP_GROUP_SIZE_WARP, OP_IMPL, detail::DppCtrl::amdgcn_dpp_wave_shift1<OP_DIR_R>::opCtrl()>;

        // Rotate in element groups
        template <uint32_t ShiftDistance>
        using ShiftL16 = ShiftL<ShiftDistance, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_shift<OP_DIR_L, ShiftDistance>::opCtrl()>;

        template <uint32_t ShiftDistance>
        using ShiftR16 = ShiftR<ShiftDistance, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::DppCtrl::amdgcn_dpp_row_shift<OP_DIR_R, ShiftDistance>::opCtrl()>;


        // Shuffle variants
        template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        using Shuffle4 = CrossLaneOps::Shuffle4<Select0,
                                Select1,
                                Select2,
                                Select3,
                                OP_IMPL,
                                detail::DppCtrl::amdgcn_dpp_shuffle_4<Select0,
                                                                        Select1,
                                                                        Select2,
                                                                        Select3>::opCtrl()>;

        template <uint32_t Select0, uint32_t Select1>
        using Shuffle2 = CrossLaneOps::Shuffle2<Select0,
                                Select1,
                                OP_IMPL,
                                detail::DppCtrl::amdgcn_dpp_shuffle_4<Select0,
                                                                        Select1,
                                                                        Select0 + Properties::OP_GROUP_SIZE_2,
                                                                        Select1 + Properties::OP_GROUP_SIZE_2>::opCtrl()>;

        // Swap variants
        using Swap2 = Swap<Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::DppCtrl::amdgcn_dpp_shuffle_4<0x02, 0x03, 0x00, 0x01>::opCtrl()>;

        // clang-format on

    } // namespace DppOps

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
        using DppFunc
            = detail::amdgcn_mov_dpp<DppOp::opCtrl(), WriteRowMask, WriteBankMask, BoundCtrl>;

        // Sanity checks
        static_assert(DppOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP,
                      "DppOp must use dpp backend");
        static_assert((DppOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SHIFT)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_REVERSE)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_SWAP)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                          || (DppOp::opId() == CrossLaneOps::Properties::OP_ID_WFALL_BCAST),
                      "DppOp is unsupported");

        // Self as prev.
        template <typename DataT>
        ROCWMMA_DEVICE static DataT exec(DataT const& val)
        {
            return DppFunc::exec(val);
        }

        template <typename DataT>
        ROCWMMA_DEVICE static void exec(DataT& val)
        {
            val = DppFunc::exec(val);
        }

        // Self as prev.
        template <typename DataT>
        ROCWMMA_DEVICE static void exec(DataT& val, DataT prev)
        {
            val = DppFunc::exec(val, prev);
        }

        // Self as prev.
        template <typename DataT>
        ROCWMMA_DEVICE static DataT exec(DataT const& val, DataT prev)
        {
            return DppFunc::exec(val, prev);
        }

        // Self as prev.
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>& v)
        {
            auto it = makeVectorIterator(v).begin();

            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = DppFunc::exec(*it);
                it++;
            }
        }

        // Scalar as prev
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>& v, DataT prev)
        {
            auto it = makeVectorIterator(v).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = DppFunc::exec(*it, prev);
                it++;
            }
        }

        // Vector as prev
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static void exec(VecT<DataT, VecSize>& v, VecT<DataT, VecSize> const& prev)
        {
            auto       it  = makeVectorIterator(v).begin();
            const auto itp = makeVectorIterator(prev).begin();

            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = DppFunc::exec(*it, *itp);
                it++;
                itp++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DPP_HPP
