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
    namespace Swizzle
    {
        /**
         * \ingroup Cross_Lane_Operations
         * @{
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
        struct Driver
        {
        private:
            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto forEach(VecT<DataT, VecSize> const& src,
                                                      detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{SwizzleOp::exec(get<Idx>(src))...};
            }

        public:
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
            ROCWMMA_DEVICE static inline auto exec(DataT const& src)
            {
                return SwizzleOp::exec(src);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src)
            {
#if ROCWMMA_ARCH_GFX1102
                VecT<DataT, VecSize> result;
                auto const           itR = makeVectorIterator(src).begin();
                auto                 itW = makeVectorIterator(result).begin();

                static_assert(decltype(itR)::range() == VecSize,
                              "VecSize inconsistent with iterator range");
                static_assert(decltype(itW)::range() == VecSize,
                              "VecSize inconsistent with iterator range");

#pragma unroll
                for(uint32_t i = 0; i < VecSize; ++i, itR++, itW++)
                {
                    get<0>(*itW) = SwizzleOp::exec(get<0>(*itR));
                }

                return result;
#else
                return forEach(src, detail::Seq<VecSize>{});
#endif
            }
        };

        /// Swizzle ops interface
        // Func::exec(src0)

        // BCast variants
        template <uint32_t ElementIdx>
        using BCast32 = Driver<SwizzleImpl::Ops::BCast32<ElementIdx>>;

        template <uint32_t ElementIdx>
        using BCast16 = Driver<SwizzleImpl::Ops::BCast16<ElementIdx>>;

        template <uint32_t ElementIdx>
        using BCast8 = Driver<SwizzleImpl::Ops::BCast8<ElementIdx>>;

        template <uint32_t ElementIdx>
        using BCast4 = Driver<SwizzleImpl::Ops::BCast4<ElementIdx>>;

        template <uint32_t ElementIdx>
        using BCast2 = Driver<SwizzleImpl::Ops::BCast2<ElementIdx>>;

        // Reverse variants
        using Reverse32 = Driver<SwizzleImpl::Ops::Reverse32>;
        using Reverse16 = Driver<SwizzleImpl::Ops::Reverse16>;
        using Reverse8  = Driver<SwizzleImpl::Ops::Reverse8>;
        using Reverse4  = Driver<SwizzleImpl::Ops::Reverse4>;
        using Reverse2  = Driver<SwizzleImpl::Ops::Reverse2>;

        // Rotate variants
        template <uint32_t RotateDistance>
        using RotateL32 = Driver<SwizzleImpl::Ops::RotateL32<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateL16 = Driver<SwizzleImpl::Ops::RotateL16<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateL8 = Driver<SwizzleImpl::Ops::RotateL8<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateL4 = Driver<SwizzleImpl::Ops::RotateL4<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateL2 = Driver<SwizzleImpl::Ops::RotateL2<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateR32 = Driver<SwizzleImpl::Ops::RotateR32<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateR16 = Driver<SwizzleImpl::Ops::RotateR16<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateR8 = Driver<SwizzleImpl::Ops::RotateR8<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateR4 = Driver<SwizzleImpl::Ops::RotateR4<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateR2 = Driver<SwizzleImpl::Ops::RotateR2<RotateDistance>>;

        // Shuffle variants
        template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        using Shuffle4 = Driver<SwizzleImpl::Ops::Shuffle4<Select0, Select1, Select2, Select3>>;

        template <uint32_t Select0, uint32_t Select1>
        using Shuffle2 = Driver<SwizzleImpl::Ops::Shuffle2<Select0, Select1>>;

        // Swap variants
        using Swap16 = Driver<SwizzleImpl::Ops::Swap16>;
        using Swap8  = Driver<SwizzleImpl::Ops::Swap8>;
        using Swap4  = Driver<SwizzleImpl::Ops::Swap4>;
        using Swap2  = Driver<SwizzleImpl::Ops::Swap2>;

        // Fft variants
        template <uint32_t SubGroupSize, uint32_t FftCtrl>
        using Fft = Driver<SwizzleImpl::Ops::Fft<SubGroupSize, FftCtrl>>;
        /** @}*/

    } // namespace Swizzle

} // namespace rocwmma

#endif // ROCWMMA_SWIZZLE_HPP
