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

// clang-format off
#include "cross_lane_ops.hpp"
#include "blend_impl.hpp"
#include "vector.hpp"
// clang-format on

namespace rocwmma
{
    namespace Blend
    {

        template <typename BlendOp>
        struct Driver
        {
        private:
            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto
                forEach(VecT<DataT, VecSize> const& src0, DataT const& src1, detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{BlendOp::exec(get<Idx>(src0), src1)...};
            }

            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto forEach(VecT<DataT, VecSize> const& src0,
                                                      VecT<DataT, VecSize> const& src1,
                                                      detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{BlendOp::exec(get<Idx>(src0), get<Idx>(src1))...};
            }

        public:
            // Sanity checks
            static_assert((BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VPERM)
                              || (BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VBLEND),
                          "BlendOp must use vperm or blend backend");
            static_assert((BlendOp::opId() == CrossLaneOps::Properties::OP_ID_BLEND)
                              || (BlendOp::opId() == CrossLaneOps::Properties::OP_ID_PERM_BYTE),
                          "BlendOp is unsupported");

            template <typename DataT>
            ROCWMMA_DEVICE static inline auto exec(DataT const& src0, DataT const& src1)
            {
                return BlendOp::exec(src0, src1);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   DataT const&                src1)
            {
                return forEach(src0, src1, detail::Seq<VecSize>{});
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   VecT<DataT, VecSize> const& src1)
            {
                return forEach(src0, src1, detail::Seq<VecSize>{});
            }
        };

        /// Blend ops interface
        // Func::exec(src0, src1)

        // Zip functions
        using ZipByte = Driver<BlendImpl::Ops::ZipByte>;
        using ZipWord = Driver<BlendImpl::Ops::ZipWord>;
        using Zip1    = Driver<BlendImpl::Ops::Zip1>;
        using Zip2    = Driver<BlendImpl::Ops::Zip2>;
        using Zip4    = Driver<BlendImpl::Ops::Zip4>;
        using Zip8    = Driver<BlendImpl::Ops::Zip8>;
        using Zip16   = Driver<BlendImpl::Ops::Zip16>;

        // Unpack functions
        using UnpackByteLo   = Driver<BlendImpl::Ops::UnpackByteLo>;
        using UnpackByteHi   = Driver<BlendImpl::Ops::UnpackByteHi>;
        using UnpackWordLo   = Driver<BlendImpl::Ops::UnpackWordLo>;
        using UnpackWordHi   = Driver<BlendImpl::Ops::UnpackWordHi>;
        using UnpackByteLoHi = Driver<BlendImpl::Ops::UnpackByteLoHi>;

    } // namespace Blend

} // namespace rocwmma

#endif // ROCWMMA_BLEND_HPP
