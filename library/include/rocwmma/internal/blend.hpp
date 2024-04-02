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
                // Vectorize to B32.
                // This way we can support B64+ types
                using B32VecT = VecT<uint32_t, sizeof(DataT) / sizeof(uint32_t)>;

                // Ensure that we can vectorize to B32
                static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(DataT), "Unable to vectorize DataT");

                // Forward to vectorized function
                auto result = exec(reinterpret_cast<B32VecT const&>(src0),
                                   reinterpret_cast<B32VecT const&>(src1));

                // Restore result to input type
                return reinterpret_cast<DataT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   DataT const&                src1)
            {
                // Reinterpret vector as B32 so we can support B64+ elements.
                constexpr uint32_t B32VecSize = sizeof(DataT) / sizeof(uint32_t) * VecSize;
                using B32VecT                 = VecT<uint32_t, B32VecSize>;
                using InputVecT               = VecT<DataT, VecSize>;

                // Ensure that we can vectorize src0 to B32
                static_assert(sizeof(InputVecT) % sizeof(uint32_t) == 0,
                              "VecT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(InputVecT),
                              "Unable to vectorize src0 to B32");

                // Reinterpret scalar as B32 vector so we can support B64+ elements.
                constexpr uint32_t B32ScalarVecSize = sizeof(DataT) / sizeof(uint32_t);
                using B32ScalarVecT                 = VecT<uint32_t, B32ScalarVecSize>;

                // Ensure that we can vectorize src1 to B32
                static_assert(sizeof(B32ScalarVecT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32ScalarVecT) == sizeof(DataT), "Unable to vectorize src1");

                auto op = [](auto&& idx, auto&& v0, auto&& s, auto&& opCtrl) {
                    // Pair up the b32 vector elements with the appropriate b32 scalar elements.
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return BlendOp::exec(get<i>(v0), get<i % B32ScalarVecSize>(s), opCtrl);
                };

                // Static unroll with cached opCtrl
                auto result = vector_generator<uint32_t, B32VecSize>()(
                    op,
                    reinterpret_cast<B32VecT const&>(src0),
                    reinterpret_cast<B32ScalarVecT const&>(src1),
                    BlendOp::opCtrl());

                // Restore result to input type
                return reinterpret_cast<InputVecT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   VecT<DataT, VecSize> const& src1)
            {
                // Reinterpret vectors as B32 so we can support B64+ elements.
                constexpr uint32_t B32VecSize = sizeof(DataT) / sizeof(uint32_t) * VecSize;
                using B32VecT                 = VecT<uint32_t, B32VecSize>;
                using InputVecT               = VecT<DataT, VecSize>;
                static_assert(sizeof(InputVecT) % sizeof(uint32_t) == 0,
                              "VecT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(InputVecT),
                              "Unable to vectorize src0 / src1 to B32");

                auto op = [](auto&& idx, auto&& v0, auto&& v1, auto&& opCtrl) {
                    // Pair up the b32 vector elements
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return BlendOp::exec(get<i>(v0), get<i>(v1), opCtrl);
                };

                // Static unroll with cached opCtrl
                auto result = vector_generator<uint32_t, B32VecSize>()(
                    op,
                    reinterpret_cast<B32VecT const&>(src0),
                    reinterpret_cast<B32VecT const&>(src1),
                    BlendOp::opCtrl());

                // Restore result to input type
                return reinterpret_cast<InputVecT&>(result);
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
        using Zip32   = Driver<BlendImpl::Ops::Zip32>;

        // Unpack functions
        using UnpackByteLo     = Driver<BlendImpl::Ops::UnpackByteLo>;
        using UnpackByteHi     = Driver<BlendImpl::Ops::UnpackByteHi>;
        using UnpackWordLo     = Driver<BlendImpl::Ops::UnpackWordLo>;
        using UnpackWordHi     = Driver<BlendImpl::Ops::UnpackWordHi>;
        using UnpackByteLoHi   = Driver<BlendImpl::Ops::UnpackByteLoHi>;
        using UnpackByte3BCast = Driver<BlendImpl::Ops::UnpackByte3BCast>;

        // Extract functions
        using ExtractByteEven = Driver<BlendImpl::Ops::ExtractByteEven>;
        using ExtractByteOdd  = Driver<BlendImpl::Ops::ExtractByteOdd>;
        using ExtractWordEven = Driver<BlendImpl::Ops::ExtractWordEven>;
        using ExtractWordOdd  = Driver<BlendImpl::Ops::ExtractWordOdd>;

        using ExtractByteEvenOdd = Driver<BlendImpl::Ops::ExtractByteEvenOdd>;
        using ExtractWordEvenOdd = Driver<BlendImpl::Ops::ExtractWordEvenOdd>;

        using ExtractByteOddEven = Driver<BlendImpl::Ops::ExtractByteOddEven>;
        using ExtractWordOddEven = Driver<BlendImpl::Ops::ExtractWordOddEven>;

        // Extract functions
        using ExtractByteEven = Driver<BlendImpl::Ops::ExtractByteEven>;
        using ExtractByteOdd  = Driver<BlendImpl::Ops::ExtractByteOdd>;
        using ExtractWordEven = Driver<BlendImpl::Ops::ExtractWordEven>;
        using ExtractWordOdd  = Driver<BlendImpl::Ops::ExtractWordOdd>;

        using ExtractByteEvenOdd = Driver<BlendImpl::Ops::ExtractByteEvenOdd>;
        using ExtractWordEvenOdd = Driver<BlendImpl::Ops::ExtractWordEvenOdd>;

        using ExtractByteOddEven = Driver<BlendImpl::Ops::ExtractByteOddEven>;
        using ExtractWordOddEven = Driver<BlendImpl::Ops::ExtractWordOddEven>;

    } // namespace Blend

} // namespace rocwmma

#endif // ROCWMMA_BLEND_HPP
