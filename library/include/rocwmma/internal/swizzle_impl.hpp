/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_SWIZZLE_IMPL_HPP
#define ROCWMMA_SWIZZLE_IMPL_HPP

#include "swizzle.hpp"
#include "utils.hpp"

namespace rocwmma
{

    //
    namespace detail
    {
        // Ctrl generators
        namespace SwizzleCtrl
        {
            template <uint32_t FftCtrl>
            struct amdgcn_swizzle_fft
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0xe000 = FFT mode
                    // FFT_CTRL: a 5-bit code for fft transform (see ISA details)
                    SWIZZLE_MODE = 0xe000,
                    FFT_CTRL     = FftCtrl & 0x1F,

                    SWIZZLE_CTRL = SWIZZLE_MODE | FFT_CTRL
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t RotationDir, uint32_t RotationDist, uint32_t GroupSize>
            struct amdgcn_swizzle_rotate
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0xc000 = rotation mode
                    // Rotation dir [10]: 0 = left, 1 = right
                    // Rotation distance [9:5]: rotation distance in element count
                    // Group size mask [4:0]: rotation within specified group size
                    // 0x00 = 32, 0x10 = 16, 0x18 = 8, 0x1C = 4 0x1E = 2, 0x1F = 1
                    SWIZZLE_MODE  = 0xc000,
                    ROTATION_DIR  = (RotationDir & 0x1) << 10,
                    ROTATION_DIST = (RotationDist & 0x1F) << 5,
                    GROUP_SIZE    = ((32u - GroupSize) & 0x1F),

                    SWIZZLE_CTRL = SWIZZLE_MODE | ROTATION_DIR | ROTATION_DIST | GROUP_SIZE
                };

                static_assert(Log2<Traits::GROUP_SIZE>::value, "GroupSize must be a power of 2");
                static_assert(Traits::ROTATION_DIR <= 1u,
                              "Rotation dir must be either 0: left or 1: right");

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct amdgcn_swizzle_shuffle_4
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0x8000 = full data shuffle within thread groups of 4
                    // Valid index selection is 0, 1, 2 or 3
                    // For every group of 4 threads, shuffle element selects:
                    SWIZZLE_MODE = 0x8000,
                    SELECT_0     = (Select0 & 0x3),
                    SELECT_1     = (Select1 & 0x3) << 2,
                    SELECT_2     = (Select2 & 0x3) << 4,
                    SELECT_3     = (Select3 & 0x3) << 6,

                    SWIZZLE_CTRL = SWIZZLE_MODE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t XorMask, uint32_t OrMask, uint32_t AndMask>
            struct amdgcn_swizzle_manual
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0x0000 = manual assignment of xor, or and and masks
                    // Note: Limited data sharing within 32 groups of threads.
                    // XorMask [14:10]
                    // OrMask [9:5]
                    // AndMask [0:4]
                    SWIZZLE_MODE = 0x0000,
                    XOR_MASK     = (XorMask & 0x1F) << 10,
                    OR_MASK      = (OrMask & 0x1F) << 5,
                    AND_MASK     = (AndMask & 0x1F),

                    SWIZZLE_CTRL = SWIZZLE_MODE | XOR_MASK | OR_MASK | AND_MASK
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

        } // namespace SwizzleCtrl

        template <typename DataT, uint32_t SwizzleCtrl>
        struct amdgcn_swizzle
        {
            __device__ static inline DataT exec(DataT input)
            {
                return reinterpret_cast<int32_t&>(input) = __builtin_amdgcn_ds_swizzle(
                           reinterpret_cast<int32_t const&>(input), SwizzleCtrl);
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_SWIZZLE_IMPL_HPP
