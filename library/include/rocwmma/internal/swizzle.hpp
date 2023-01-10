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
    namespace SwizzleOps
    {
        /**
         * \ingroup Cross-Lane Operations
         * \defgroup Swizzle Ops
         *
         * @brief Cross-lane operations implemented with the amdgcn_ds_swizzle backend.
         */

        // clang-format off

        using CrossLaneOps::Properties;
        using CrossLaneOps::BCast;
        using CrossLaneOps::Reverse;
        using CrossLaneOps::RotateR;
        using CrossLaneOps::RotateL;
        using CrossLaneOps::Swap;

        constexpr uint32_t OP_IMPL  = Properties::OP_IMPL_SWIZZLE;
        constexpr uint32_t OP_DIR_L = Properties::OP_DIR_L;
        constexpr uint32_t OP_DIR_R = Properties::OP_DIR_R;
        constexpr uint32_t OP_ID_FFT = Properties::OP_ID_FFT;

        // BCast variants
        template <uint32_t ElementIdx>
        using BCast32 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_32, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x00, ElementIdx, 0x00>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast16 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x00, ElementIdx, 0x10>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast8 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x00, ElementIdx, 0x18>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast4 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x00, ElementIdx, 0x1C>::opCtrl()>;

        template <uint32_t ElementIdx>
        using BCast2 = BCast<ElementIdx, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x00, ElementIdx, 0x1E>::opCtrl()>;

        // Reverse variants
        using Reverse32 = Reverse<Properties::OP_GROUP_SIZE_32, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x1F, 0x00, 0x1F>::opCtrl()>;

        using Reverse16 = Reverse<Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x0F, 0x00, 0x1F>::opCtrl()>;

        using Reverse8 = Reverse<Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x07, 0x00, 0x1F>::opCtrl()>;

        using Reverse4 = Reverse<Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x03, 0x00, 0x1F>::opCtrl()>;

        using Reverse2 = Reverse<Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x01, 0x00, 0x1F>::opCtrl()>;

        // RotateL variants
        template <uint32_t RotateDistance>
        using RotateL32 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_32, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_L, RotateDistance, 32u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL16 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_L, RotateDistance, 16u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL8 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_L, RotateDistance, 8u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL4 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_L, RotateDistance, 4u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateL2 = RotateL<RotateDistance, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_L, RotateDistance, 2u>::opCtrl()>;

        // RotateR variants
        template <uint32_t RotateDistance>
        using RotateR32 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_32, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_R, RotateDistance, 32u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR16 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_R, RotateDistance, 16u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR8 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_R, RotateDistance, 8u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR4 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_R, RotateDistance, 4u>::opCtrl()>;

        template <uint32_t RotateDistance>
        using RotateR2 = RotateR<RotateDistance, Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_rotate<OP_DIR_R, RotateDistance, 2u>::opCtrl()>;

        // Shuffle variants
        template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        using Shuffle4 = CrossLaneOps::Shuffle4<Select0,
                                                Select1,
                                                Select2,
                                                Select3,
                                                OP_IMPL,
                                                detail::SwizzleCtrl::amdgcn_swizzle_shuffle_4<Select0, Select1, Select2, Select3>::opCtrl()>;

        template <uint32_t Select0, uint32_t Select1>
        using Shuffle2 = CrossLaneOps::Shuffle2<Select0,
                                                Select1,
                                                OP_IMPL,
                                                detail::SwizzleCtrl::amdgcn_swizzle_shuffle_4<Select0, Select1, 2u + Select0, 2u + Select1>::opCtrl()>;

        // Swap variants
        using Swap16 = Swap<Properties::OP_GROUP_SIZE_16, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x10, 0x00, 0x1F>::opCtrl()>;

        using Swap8 = Swap<Properties::OP_GROUP_SIZE_8, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x08, 0x00, 0x1F>::opCtrl()>;

        using Swap4 = Swap<Properties::OP_GROUP_SIZE_4, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x04, 0x00, 0x1F>::opCtrl()>;

        using Swap2 = Swap<Properties::OP_GROUP_SIZE_2, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_manual<0x02, 0x00, 0x1F>::opCtrl()>;

        /*! \class Fft
        *  \brief Supports FFT-like cross-bar transforms
        *
        * @tparam FftCtrl - 5-bit swizzle code (see instruction ISA for layouts)
        */
        template <uint32_t SubGroupSize, uint32_t FftCtrl>
        using Fft = CrossLaneOps::Fft<SubGroupSize, FftCtrl, OP_IMPL, detail::SwizzleCtrl::amdgcn_swizzle_fft<FftCtrl>::opCtrl()>;

        // clang-format on

    } // namespace SwizzleOps

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
        using SwizzleFunc = detail::amdgcn_swizzle<SwizzleOp::opCtrl()>;

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
        ROCWMMA_DEVICE static void exec(DataT& v)
        {
            v = SwizzleFunc::exec(v);
        }

        template <typename DataT>
        ROCWMMA_DEVICE static DataT exec(DataT const& v)
        {
            return SwizzleFunc::exec(v);
        }

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
                *it = SwizzleFunc::exec(*it);
                it++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_SWIZZLE_HPP
