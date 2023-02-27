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
#ifndef ROCWMMA_BLEND_IMPL_HPP
#define ROCWMMA_BLEND_IMPL_HPP

#include "blend.hpp"

namespace rocwmma
{

    namespace detail
    {
        template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        struct amdgcn_blend_byte
        {
        private:
            enum Traits : uint32_t
            {
                // Byte blend mode: for each 32b element, select the 8b ordering.
                // 0u <= select_index < 4u  = src0
                // 4u <= select_index < 8u  = src1
                //
                // dst[7:0]   = SELECT_BYTE_0{ concat(src1[31:0], src0[31:0]) }
                // dst[15:8]  = SELECT_BYTE_1{ concat(src1[31:0], src0[31:0]) }
                // dst[24:16] = SELECT_BYTE_2{ concat(src1[31:0], src0[31:0]) }
                // dst[31:23] = SELECT_BYTE_3{ concat(src1[31:0], src0[31:0]) }
                BLEND_MODE = 0x0,
                SELECT_0   = (Select0 & 0xFF),
                SELECT_1   = (Select1 & 0xFF) << 8,
                SELECT_2   = (Select2 & 0xFF) << 16,
                SELECT_3   = (Select3 & 0xFF) << 24,

                BLEND_CTRL = BLEND_MODE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
            };

        public:
            constexpr static uint32_t opCtrl()
            {
                return Traits::BLEND_CTRL;
            }
        };

        template <uint32_t GroupSize>
        struct amdgcn_blend_zip_b32
        {
        private:
            enum Traits : uint32_t
            {
                MASK_BASE = LsbMask<32>::value
            };

        public:
            constexpr static uint32_t maskCtrl()
            {
                // Just like a zipper, alternate mask between src0 (0x000000000)
                // and src1 (0xFFFFFFFF) based on threadIdx.x.
                // GroupSize of N means that N elements are used from src0, followed
                // by N elements from src1, and so on.
                return ((threadIdx.x >> Log2<GroupSize>::value) & 0x1) * MASK_BASE;
            }
        };

        /*! \class amdgcn_perm
        *  \brief Implements the element blending backend.
        * This backend provides sub-b32 permutation between two sources. Blending in this
        * context means that ordered pairs of 32b elements (src0, src1) may be permuted to
        * generate a 32b element result for every lane.
        *
        * E.g. Byte-wise permutation of ordered source elements:
        * Result[31:0] = Src0[7:0],Src0[7:0],Src1[15:8],Src1[23:16]
        *
        * @tparam BlendCtrl is the control code for byte blending between src0 and src1
        * @arg src0 is the 'lower' reference
        * @arg src1 is the 'upper' reference
        * @arg opCtrl is mask selection for each byte
        */
        template <uint32_t BlendCtrl>
        struct amdgcn_perm
        {
            template <typename DataT>
            ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1)
            {
                // NOTE: src0 and src1 are flipped here due to spec's select
                // concatenation of i[3:0] = src1 and i[7:4] = src0 .
                // amdgcn_blend_byte does the inverse of this to make
                // the rocWMMA interface more intuitive, with src0 as the lower
                // 4 bytes and src1 in the upper 4 bytes.
                reinterpret_cast<uint32_t&>(src0)
                    = __builtin_amdgcn_perm(reinterpret_cast<uint32_t&>(src1),
                                            reinterpret_cast<uint32_t&>(src0),
                                            BlendCtrl);
                return src0;
            }
        };

        /*! \class amdgcn_blend
        *  \brief Implements the bitwise blend backend between 32b elements of two src vectors.
        * Blend means bit-wise ordered combination of two source vector elements, in a mutally exclusive
        * fashion, as in either / or, without permutation.
        *
        * E.g. res = src0 <= mask = 0x00000000
        *      res = src1 <= mask = 0xFFFFFFFF
        *      res = select_bits(src0, src1) <= mask 0's take bits of src0 and mask1's take bits of src1
        *
        * @arg src0 is the 'lower' source
        * @arg src1 is the 'upper' source
        * @arg mask is a 32b mask. Lo bits (0) select from src0, hi bits (1) select
        * from src1.
        */
        struct amdgcn_blend
        {
            template <typename DataT>
            ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1, uint32_t mask)
            {
                return (src1 & mask) | (src0 & ~mask);
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_BLEND_IMPL_HPP
