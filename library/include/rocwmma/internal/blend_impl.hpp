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

        /*! \class amdgcn_perm
        *  \brief Implements the blending backend, not to be mislead by the function name.
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

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_BLEND_IMPL_HPP
