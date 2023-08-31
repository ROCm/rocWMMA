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
    namespace Dpp
    {
        /**
         * \ingroup Cross_Lane_Operations
         *
         * @brief Cross-lane operations implemented with the amdgcn_mov_dpp backend.
         * @{
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
        struct Driver
        {
        private:
            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto forEach(VecT<DataT, VecSize> const& src,
                                                      detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{
                    DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(get<Idx>(src),
                                                                                 get<Idx>(src))...};
            }

            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto
                forEach(VecT<DataT, VecSize> const& src0, DataT const& src1, detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                static_assert(sizeof(DataT) == sizeof(uint32_t), "Scalar must be 32b");
                return VecT<DataT, VecSize>{
                    DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(get<Idx>(src0),
                                                                                 src1)...};
            }

            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto forEach(VecT<DataT, VecSize> const& src0,
                                                      VecT<DataT, VecSize> const& src1,
                                                      detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{
                    DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(
                        get<Idx>(src0), get<Idx>(src1))...};
            }

        public:
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
            ROCWMMA_DEVICE static inline auto exec(DataT const& src0, DataT const& src1)
            {
                return DppOp::template exec<WriteRowMask, WriteBankMask, BoundCtrl>(src0, src1);
            }

            // Self as prev
            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src)
            {
                return forEach(src, detail::Seq<VecSize>{});
            }

            // Scalar as prev
            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   DataT const&                src1)
            {
                return forEach(src0, src1, detail::Seq<VecSize>{});
            }

            // Vector as prev
            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   VecT<DataT, VecSize> const& src1)
            {
                return forEach(src0, src1, detail::Seq<VecSize>{});
            }
        };

        /// Dpp ops interface
        // Func::exec(src0)

        // BCast variants
        template <uint32_t ElementIdx,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using BCast16 = Driver<DppImpl::Ops::BCast16<ElementIdx>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t ElementIdx,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using BCast4 = Driver<DppImpl::Ops::BCast4<ElementIdx>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t ElementIdx,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using BCast2 = Driver<DppImpl::Ops::BCast2<ElementIdx>, RowMask, BankMask, BoundCtrl>;

        // Special BCast variants:
        // BCast<M>x<N>, where:
        // <M> = subgroup size
        // <N> = element idx
        // NOTE: These functions only broadcast the <N>th element of the current subgroup to the NEXT subgroup
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using BCast16x15 = Driver<DppImpl::Ops::BCast16x15, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using BCast32x31 = Driver<DppImpl::Ops::BCast32x31, RowMask, BankMask, BoundCtrl>;

        // Move variants
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using MaskMove = Driver<DppImpl::Ops::MaskMove, RowMask, BankMask, BoundCtrl>;

        // Reversal variants
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using Reverse16 = Driver<DppImpl::Ops::Reverse16, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using Reverse8 = Driver<DppImpl::Ops::Reverse8, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using Reverse4 = Driver<DppImpl::Ops::Reverse4, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using Reverse2 = Driver<DppImpl::Ops::Reverse2, RowMask, BankMask, BoundCtrl>;

        // Rotation variants
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using RotateWaveR1 = Driver<DppImpl::Ops::RotateWaveR1, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using RotateWaveL1 = Driver<DppImpl::Ops::RotateWaveL1, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RotateDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using RotateR16
            = Driver<DppImpl::Ops::RotateR16<RotateDistance>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RotateDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using RotateL4
            = Driver<DppImpl::Ops::RotateL4<RotateDistance>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RotateDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using RotateR4
            = Driver<DppImpl::Ops::RotateR4<RotateDistance>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RotateDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using RotateL2
            = Driver<DppImpl::Ops::RotateL2<RotateDistance>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RotateDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using RotateR2
            = Driver<DppImpl::Ops::RotateR2<RotateDistance>, RowMask, BankMask, BoundCtrl>;

        // Shift variants
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using ShiftWaveL1 = Driver<DppImpl::Ops::ShiftWaveL1, RowMask, BankMask, BoundCtrl>;

        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using ShiftWaveR1 = Driver<DppImpl::Ops::ShiftWaveR1, RowMask, BankMask, BoundCtrl>;

        template <uint32_t ShiftDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using ShiftL16
            = Driver<DppImpl::Ops::ShiftL16<ShiftDistance>, RowMask, BankMask, BoundCtrl>;

        template <uint32_t ShiftDistance,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using ShiftR16
            = Driver<DppImpl::Ops::ShiftR16<ShiftDistance>, RowMask, BankMask, BoundCtrl>;

        // Shuffle variants
        template <uint32_t Select0,
                  uint32_t Select1,
                  uint32_t Select2,
                  uint32_t Select3,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using Shuffle4 = Driver<DppImpl::Ops::Shuffle4<Select0, Select1, Select2, Select3>,
                                RowMask,
                                BankMask,
                                BoundCtrl>;

        template <uint32_t Select0,
                  uint32_t Select1,
                  uint32_t RowMask   = 0xF,
                  uint32_t BankMask  = 0xF,
                  bool     BoundCtrl = false>
        using Shuffle2
            = Driver<DppImpl::Ops::Shuffle2<Select0, Select1>, RowMask, BankMask, BoundCtrl>;

        // Swap variants
        template <uint32_t RowMask = 0xF, uint32_t BankMask = 0xF, bool BoundCtrl = false>
        using Swap2 = Driver<DppImpl::Ops::Swap2, RowMask, BankMask, BoundCtrl>;

        using Nop = MaskMove<>;
        /** @}*/

    } // namespace Dpp

} // namespace rocwmma

#endif // ROCWMMA_DPP_HPP
