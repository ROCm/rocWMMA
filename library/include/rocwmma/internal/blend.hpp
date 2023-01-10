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
    namespace BlendOps
    {
        /**
         * \ingroup Cross-Lane Operations
         * \defgroup Blend Ops
         *
         * @brief Cross-lane operations implemented with the amdgcn_perm backend.
         *
         * Here we build out the cross-lane properties specific to blending, such as the backend (OP_IMPL_BLEND).
         *
         * These definitions are for blend support, so we specify this backend here. These ops must
         * inherit the meta-data front end from CrossLaneOps.
         */

        // clang-format off

        using CrossLaneOps::Properties;
        using CrossLaneOps::Blend4;

        constexpr uint32_t OP_IMPL  = Properties::OP_IMPL_VPERM;

        // For each 32b element, blend bytes from src0 and src1
        template<uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        using ByteBlend = Blend4<Select0, Select1, Select2, Select3, OP_IMPL, detail::amdgcn_blend_byte<Select0, Select1, Select2, Select3>::opCtrl()>;

        // For each 32b element, blend short(16b) from src0 and src1
        template<uint32_t Select0, uint32_t Select1>
        using WordBlend = Blend4<Select0 * 2u, Select0 * 2u + 1u, Select1 * 2u, Select1 * 2u + 1, OP_IMPL,
                      detail::amdgcn_blend_byte<Select0 * 2u, Select0 * 2u + 1u, Select1 * 2u, Select1 * 2u + 1>::opCtrl()>;

        // Blend even bytes from src0 and odd bytes from src1
        using ZipByte = ByteBlend<0u, 5u, 2u, 7u>;
        using ZipByteR = ByteBlend<4u, 1u, 6u, 3u>;

        // Blend even words from src0 and odd bytes from src1
        using ZipWord = WordBlend<0u, 3u>;
        using ZipWordR = WordBlend<2u, 1u>;

        // Alternate lo bytes between src0 and src1
        using UnpackByteLo = ByteBlend<0u, 4u, 1u, 5u>;
        using UnpackByteLoR = ByteBlend<4u, 0u, 5u, 1u>;

        // Alternate hi bytes between src0 and src1
        using UnpackByteHi = ByteBlend<2u, 6u, 3u, 7u>;
        using UnpackByteHiR = ByteBlend<6u, 2u, 7u, 3u>;

        // Alternate lo words between src0 and src1
        using UnpackWordLo = WordBlend<0u, 2u>;
        using UnpackWordLoR = WordBlend<2u, 0u>;

        // Alternate hi words between src0 and src1
        using UnpackWordHi = WordBlend<1u, 3u>;
        using UnpackWordHiR = WordBlend<3u, 1u>;

        // Alternate lo bytes from src0 and hi bytes from src1
        using UnpackByteLoHi = ByteBlend<0u, 6u, 1u, 7u>;
        using UnpackByteLoHiR = ByteBlend<4u, 2u, 5u, 3u>;

        // clang-format on
    }

    template <typename BlendOp>
    struct Blend
    {
        using BlendFunc = detail::amdgcn_perm<BlendOp::opCtrl()>;

        // Sanity checks
        static_assert(BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VPERM,
                      "BlendOp must use vperm backend");
        static_assert(BlendOp::opId() == CrossLaneOps::Properties::OP_ID_BLEND,
                      "BlendOp is unsupported");

        template <typename DataT>
        ROCWMMA_HOST_DEVICE static inline DataT exec(DataT const& src0, DataT const& src1)
        {
            return BlendFunc::exec(src0, src1);
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_HOST_DEVICE static void exec(VecT<DataT, VecSize>& src0, DataT const& src1)
        {
            auto it = makeVectorIterator(src0).begin();
            static_assert(decltype(it)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it = BlendFunc::exec(*it, src1);
                it++;
            }
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_HOST_DEVICE static void exec(VecT<DataT, VecSize>&       src0,
                                             VecT<DataT, VecSize> const& src1)
        {
            auto it0 = makeVectorIterator(src0).begin();
            auto it1 = makeVectorIterator(src1).begin();
            static_assert(decltype(it0)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it0 = BlendFunc::exec(*it0, *it1);
                it0++;
                it1++;
            }
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_HOST_DEVICE static void exec(VecT<DataT, VecSize>& src0)
        {
            auto it0 = makeVectorIterator(src0).begin();
            static_assert(decltype(it0)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *it0 = BlendFunc::exec(*it0, *it0);
                it0++;
            }
        }

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_HOST_DEVICE static auto exec(VecT<DataT, VecSize> const& src0)
        {
            VecT<DataT, VecSize> result;
            auto const           itR = makeVectorIterator(src0).begin();
            auto                 itW = makeVectorIterator(result).begin();
            static_assert(decltype(itR)::range() == VecSize,
                          "VecSize inconsistent with iterator range");

            // Loop through entire vector
#pragma unroll
            for(uint32_t i = 0; i < VecSize; ++i)
            {
                *itW = BlendFunc::exec(*itR, *itR);
                itR++;
                itW++;
            }

            return result;
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_BLEND_HPP
