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
#ifndef ROCWMMA_MOVE_DPP_IMPL_HPP
#define ROCWMMA_MOVE_DPP_IMPL_HPP

#include "dpp.hpp"

namespace rocwmma
{

    namespace detail
    {
        // Ctrl generators
        namespace DppCtrl
        {

            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct amdgcn_dpp_shuffle_4
            {
            private:
                enum Traits : uint32_t
                {
                    // Valid index selection is 0, 1, 2 or 3
                    // For every group of 4 threads, shuffle element selects:
                    DPP_BASE = 0x0,
                    SELECT_0 = (Select0 & 0x3),
                    SELECT_1 = (Select1 & 0x3) << 2,
                    SELECT_2 = (Select2 & 0x3) << 4,
                    SELECT_3 = (Select3 & 0x3) << 6,

                    DPP_CTRL = DPP_BASE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            // This 'nop' is to signify that the opctrl will have no effect.
            // Used in architectures that don't support particular ops.
            // Still respects masking and bound control flags.
            using amdgcn_dpp_nop = amdgcn_dpp_shuffle_4<0u, 1u, 2u, 3u>;

            template <uint32_t ShiftDir, uint32_t ShiftDist>
            struct amdgcn_dpp_row_shift
            {
            private:
                enum Traits : uint32_t
                {
                    // Shift mode base offset: 0x0100 (undefined for ShiftDistance = 0)
                    // Shift dir [4] : 0 = left, 1 = right
                    // Shift distance [3:0]: shift distance in element count
                    DPP_BASE   = 0x0100,
                    SHIFT_DIR  = (ShiftDir & 0x1) << 4,
                    SHIFT_DIST = (ShiftDist & 0xF),

                    DPP_CTRL = DPP_BASE | SHIFT_DIR | SHIFT_DIST
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDist>
            struct amdgcn_dpp_row_rotate_r
            {
            private:
                enum Traits : uint32_t
                {
                    // Rotate mode base offset: 0x0120
                    // Rotate distance [3:0]: rotate distance in element count
                    DPP_BASE    = 0x0120,
                    ROTATE_DIST = (RotateDist & 0xF),

                    DPP_CTRL = DPP_BASE | ROTATE_DIST
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct amdgcn_dpp_row_reverse
            {
            private:
                enum Traits : uint32_t
                {
                    // Reverse mode base offset: 0x0140
                    DPP_CTRL = 0x0140,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct amdgcn_dpp_half_row_reverse
            {
            private:
                enum Traits : uint32_t
                {
                    // Reverse mode base offset: 0x0140
                    DPP_CTRL = 0x0141,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

// GPU exclusion for unsupported targets, but assume host is valid
// for testing purposes.
#if !__gfx908__ // + Host

            template <uint32_t ElementIdx>
            struct amdgcn_dpp_row_bcast
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0150
                    DPP_CTRL = 0x0150 + ElementIdx,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

#else // __gfx908__

            template <uint32_t ElementIdx>
            struct amdgcn_dpp_row_bcast
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = amdgcn_dpp_nop::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast is not supported on gfx908")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

#endif // !__gfx908__

// GPU exclusion for unsupported targets, but assume host is valid
// for testing purposes.
#if !__gfx1100__ && !__gfx1101__ && !__gfx1102__ // + Host

            template <uint32_t ShiftDir>
            struct amdgcn_dpp_wave_shift1
            {
            private:
                enum Traits : uint32_t
                {
                    // Shift mode base offset: 0x0130
                    // Shift dir [3] : 0 = left, 1 = right
                    // wave_shift1_L = 0x0130
                    // wave_shift1_R = 0x0138
                    DPP_BASE  = 0x0130,
                    SHIFT_DIR = (ShiftDir & 0x1) << 3,

                    DPP_CTRL = DPP_BASE | SHIFT_DIR
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDir>
            struct amdgcn_dpp_wave_rotate1
            {
            private:
                enum Traits : uint32_t
                {
                    // Rotate mode base offset: 0x0134
                    // Rotate dir [3] : 0 = left, 1 = right
                    // wave_rotate1_L = 0x0134
                    // wave_rotate1_R = 0x013c
                    DPP_BASE   = 0x0134,
                    ROTATE_DIR = (RotateDir & 0x1) << 3,

                    DPP_CTRL = DPP_BASE | ROTATE_DIR
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct amdgcn_dpp_row_bcast15
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0142
                    DPP_CTRL = 0x0142,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct amdgcn_dpp_row_bcast31
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0143
                    DPP_CTRL = 0x0143,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

#else // __gfx1100__ || __gfx1101__ || __gfx1102__

            template <uint32_t ShiftDir>
            struct amdgcn_dpp_wave_shift1
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = amdgcn_dpp_nop::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_wave_shift1 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            template <uint32_t RotateDir>
            struct amdgcn_dpp_wave_rotate1
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = amdgcn_dpp_nop::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_wave_rotate1 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            struct amdgcn_dpp_row_bcast15
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = amdgcn_dpp_nop::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast15 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            struct amdgcn_dpp_row_bcast31
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = amdgcn_dpp_nop::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast31 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

#endif // !__gfx1100__ && !__gfx1101__ && !__gfx1102__

            // Derivatives
            template <uint32_t RotateDir, uint32_t RotateDistance>
            struct amdgcn_dpp_bank_rotate
            {
            private:
                enum Traits : uint32_t
                {
                    ROTATE_DISTANCE
                    = (RotateDir == CrossLaneOps::Properties::OP_DIR_L ? RotateDistance
                                                                       : 4u - RotateDistance),
                    SELECT_0 = (0u + ROTATE_DISTANCE) % 4u,
                    SELECT_1 = (1u + ROTATE_DISTANCE) % 4u,
                    SELECT_2 = (2u + ROTATE_DISTANCE) % 4u,
                    SELECT_3 = (3u + ROTATE_DISTANCE) % 4u,

                    DPP_CTRL
                    = amdgcn_dpp_shuffle_4<SELECT_0, SELECT_1, SELECT_2, SELECT_3>::opCtrl()
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDir, uint32_t RotateDistance>
            struct amdgcn_dpp_half_bank_rotate
            {
            private:
                enum Traits : uint32_t
                {
                    ROTATE_DISTANCE
                    = (RotateDir == CrossLaneOps::Properties::OP_DIR_L ? RotateDistance
                                                                       : 2u - RotateDistance),
                    SELECT_0 = (0u + ROTATE_DISTANCE) % 2u,
                    SELECT_1 = (1u + ROTATE_DISTANCE) % 2u,
                    SELECT_2 = 2u + SELECT_0,
                    SELECT_3 = 2u + SELECT_1,

                    DPP_CTRL
                    = amdgcn_dpp_shuffle_4<SELECT_0, SELECT_1, SELECT_2, SELECT_3>::opCtrl()
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

        } // namespace DppCtrl

        template <uint32_t DppCtrl, uint32_t WriteRowMask, uint32_t WriteBankMask, bool BoundCtrl>
        struct amdgcn_mov_dpp
        {
            template <typename DataT>
            __device__ static inline DataT exec(DataT input)
            {
                reinterpret_cast<int32_t&>(input) = __builtin_amdgcn_update_dpp(
                    reinterpret_cast<int32_t const&>(input), // use self as prev
                    reinterpret_cast<int32_t const&>(input),
                    DppCtrl, // DPP control code
                    WriteRowMask, // Mask for affected rows
                    WriteBankMask, // Mask for affected banks
                    BoundCtrl); // Fill in 0 on invalid indices
                return input;
            }

            template <typename DataT>
            __device__ static inline DataT exec(DataT input, DataT prev)
            {
                reinterpret_cast<int32_t&>(input) = __builtin_amdgcn_update_dpp(
                    reinterpret_cast<int32_t const&>(prev), // fill prev value
                    reinterpret_cast<int32_t const&>(input),
                    DppCtrl, // DPP control code
                    WriteRowMask, // Mask for affected rows
                    WriteBankMask, // Mask for affected banks
                    BoundCtrl); // Fill in 0 on invalid indices
                return input;
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_MOVE_DPP_IMPL_HPP
