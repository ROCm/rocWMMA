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
#ifndef ROCWMMA_CROSS_LANE_OPS_HPP
#define ROCWMMA_CROSS_LANE_OPS_HPP

#include "constants.hpp"

namespace rocwmma
{

    namespace CrossLaneOps
    {
        enum Properties : uint32_t
        {
            // 32b Element Operation IDs
            OP_ID_ROTATE      = 0x00, // position rotation
            OP_ID_SHIFT       = 0x01, // position shift
            OP_ID_SHUFFLE     = 0x02, // position shuffle
            OP_ID_REVERSE     = 0x03, // position mirror
            OP_ID_SWAP        = 0x04, // neighbour swap
            OP_ID_BCAST       = 0x05, // broadcast element
            OP_ID_FFT         = 0x06, // fft shuffle
            OP_ID_BLOCK_BCAST = 0x07, // broadcast block
            OP_ID_WFALL_BCAST = 0x08, // broadcast last element to next block

            // Identifiers of backend implementation
            OP_IMPL_DPP     = 0x30,
            OP_IMPL_SWIZZLE = 0x31,
            OP_IMPL_PERMUTE = 0x32,

            // Directional properties
            OP_DIR_L = 0x00, // = left  (towards LSB)
            OP_DIR_R = 0x01 // = right (towards MSB)
        };

        /*! \class OpBase
        *  \brief Container for meta-properties common to many cross-lane operations.
        *
        * @tparam OpId classification of the operation: see Properties
        * @tparam ThreadCount number of active threads per wave (wave32 or wave64)
        * @tparam SubGroupSize most operations repeat for groups of threads in the entire register
        * @tparam OpImpl backend implementation of the op: see Properties
        * @tparam OpCtrl backend-specific control code to invoke the operation.
        */
        template <uint32_t OpId,
                  uint32_t ThreadCount,
                  uint32_t SubGroupSize,
                  uint32_t OpImpl,
                  uint32_t OpCtrl>
        struct OpBase
        {
            enum : uint32_t
            {
                OP_ID      = OpId,
                OP_IMPL    = OpImpl,
                OP_CTRL    = OpCtrl,
                WAVE_SIZE  = ThreadCount,
                GROUP_SIZE = SubGroupSize,
            };

            constexpr static uint32_t opId()
            {
                return OP_ID;
            }
            constexpr static uint32_t opImpl()
            {
                return OP_IMPL;
            }
            constexpr static uint32_t opCtrl()
            {
                return OP_CTRL;
            }
            constexpr static uint32_t waveSize()
            {
                return WAVE_SIZE;
            }
            constexpr static uint32_t groupSize()
            {
                return GROUP_SIZE;
            }
        };

        /*! \class Rotate
        *  \brief Perform element-wise rotation in direction <RotateDir> in sub-groups of <SubGroupSize> threads.
        *
        * @tparam RotateDir rotation direction: see Properties
        * @tparam RotateDistance element positions to move in specified direction. Positions wrapped by sub group size.
        */
        template <uint32_t RotateDir,
                  uint32_t RotateDist,
                  uint32_t SubGroupSize,
                  uint32_t OpImpl,
                  uint32_t OpCtrl>
        struct Rotate : public OpBase<Properties::OP_ID_ROTATE,
                                      AMDGCN_WAVE_SIZE,
                                      SubGroupSize,
                                      OpImpl,
                                      OpCtrl>
        {
            enum : uint32_t
            {
                OP_DIR  = RotateDir,
                OP_DIST = RotateDist
            };

            constexpr static uint32_t opDir()
            {
                return OP_DIR;
            }
            constexpr static uint32_t opDist()
            {
                return OP_DIST;
            }
        };

        template <uint32_t RotateDistance, uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        using RotateR = Rotate<Properties::OP_DIR_R, RotateDistance, SubGroupSize, OpImpl, OpCtrl>;

        template <uint32_t RotateDistance, uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        using RotateL = Rotate<Properties::OP_DIR_L, RotateDistance, SubGroupSize, OpImpl, OpCtrl>;

        /*! \class Shift
        *  \brief Perform element-wise shift in direction <ShiftDir> in sub-groups of <SubGroupSize> threads.
        *
        * @tparam ShiftDir rotation direction: see Properties
        * @tparam ShiftDistance element positions to move in specified direction. Positions do not wrap around
        * the sub group size.
        */
        template <uint32_t ShiftDir,
                  uint32_t ShiftDist,
                  uint32_t SubGroupSize,
                  uint32_t OpImpl,
                  uint32_t OpCtrl>
        struct Shift
            : public OpBase<Properties::OP_ID_SHIFT, AMDGCN_WAVE_SIZE, SubGroupSize, OpImpl, OpCtrl>
        {
            enum : uint32_t
            {
                OP_DIR  = ShiftDir,
                OP_DIST = ShiftDist
            };

            constexpr static uint32_t opDir()
            {
                return OP_DIR;
            }
            constexpr static uint32_t opDist()
            {
                return OP_DIST;
            }
        };

        template <uint32_t ShiftDistance, uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        using ShiftR = Shift<Properties::OP_DIR_R, ShiftDistance, SubGroupSize, OpImpl, OpCtrl>;

        template <uint32_t ShiftDistance, uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        using ShiftL = Shift<Properties::OP_DIR_L, ShiftDistance, SubGroupSize, OpImpl, OpCtrl>;

        /*! \class BCast
        *  \brief Performs localized broadcast of one element in each sub-group to the entire sub-group.
        *
        * @tparam ElementIdx - element index to broadcast to rest of the sub-group
        */
        template <uint32_t ElementIdx, uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct BCast
            : public OpBase<Properties::OP_ID_BCAST, AMDGCN_WAVE_SIZE, SubGroupSize, OpImpl, OpCtrl>
        {
            enum : uint32_t
            {
                ELEMENT_IDX = ElementIdx,
            };

            constexpr static uint32_t elementIdx()
            {
                return ELEMENT_IDX;
            }
        };

        /*! \class Reverse
        *  \brief Perform reversal of elements in sub-groups of <SubGroupSize> threads.
        */
        template <uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct Reverse : public OpBase<Properties::OP_ID_REVERSE,
                                       AMDGCN_WAVE_SIZE,
                                       SubGroupSize,
                                       OpImpl,
                                       OpCtrl>
        {
        };

        /*! \class Swap
        *  \brief Perform swap of neigbouring sub-groups of <SubGroupSize> threads.
        */
        template <uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct Swap
            : public OpBase<Properties::OP_ID_SWAP, AMDGCN_WAVE_SIZE, SubGroupSize, OpImpl, OpCtrl>
        {
        };

        /*! \class Shuffle
        *  \brief Perform localized shuffling within sub-groups of <SubGroupSize> threads.
        */
        template <uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct Shuffle : public OpBase<Properties::OP_ID_SHUFFLE,
                                       AMDGCN_WAVE_SIZE,
                                       SubGroupSize,
                                       OpImpl,
                                       OpCtrl>
        {
        };

        // Common Shuffle variants
        /*! \class Shuffle<N>
        *  \brief Perform localized shuffling within all sub-groups of <N> threads.
        * <N> = group size.
        *
        * @tparam Select0 - index of element to shuffle to index 0
        * @tparam Select1 - index of element to shuffle to index 1
        * @tparam Select2 - index of element to shuffle to index 2
        * @tparam Select3 - index of element to shuffle to index 3
        */
        template <uint32_t Select0,
                  uint32_t Select1,
                  uint32_t Select2,
                  uint32_t Select3,
                  uint32_t OpImpl,
                  uint32_t OpCtrl>
        struct Shuffle4 : public Shuffle<4u, OpImpl, OpCtrl>
        {
            enum : uint32_t
            {
                SELECT_0 = Select0,
                SELECT_1 = Select1,
                SELECT_2 = Select2,
                SELECT_3 = Select3,
            };

            constexpr static uint32_t select0()
            {
                return SELECT_0;
            }
            constexpr static uint32_t select1()
            {
                return SELECT_1;
            }
            constexpr static uint32_t select2()
            {
                return SELECT_2;
            }
            constexpr static uint32_t select3()
            {
                return SELECT_3;
            }
        };

        template <uint32_t Select0, uint32_t Select1, uint32_t OpImpl, uint32_t OpCtrl>
        struct Shuffle2 : public Shuffle<2u, OpImpl, OpCtrl>
        {
            enum : uint32_t
            {
                SELECT_0 = Select0,
                SELECT_1 = Select1,
            };

            constexpr static uint32_t select0()
            {
                return SELECT_0;
            }
            constexpr static uint32_t select1()
            {
                return SELECT_1;
            }
        };

        /*! \class Fft
        *  \brief Supports FFT-like cross-bar transforms
        */
        template <uint32_t FftCtrl, uint32_t OpImpl, uint32_t OpCtrl>
        struct Fft : public OpBase<Properties::OP_ID_FFT,
                                   AMDGCN_WAVE_SIZE,
                                   AMDGCN_WAVE_SIZE,
                                   OpImpl,
                                   OpCtrl>
        {
            enum : uint32_t
            {
                FFT_CTRL = FftCtrl
            };

            constexpr static uint32_t fftCtrl()
            {
                return FFT_CTRL;
            }
        };

        /*! \class BlockBCast
        *  \brief Performs broadcast of one block of elements to all other blocks.
        *
        * @tparam BlockIdx - block index to broadcast to rest of the other blocks.
        */
        template <uint32_t BlockIdx, uint32_t BlockSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct BlockBCast : public OpBase<Properties::OP_ID_BLOCK_BCAST,
                                          AMDGCN_WAVE_SIZE,
                                          BlockSize,
                                          OpImpl,
                                          OpCtrl>
        {
            enum : uint32_t
            {
                ELEMENT_IDX = BlockIdx,
            };

            constexpr static uint32_t elementIdx()
            {
                return BlockIdx;
            }
        };

        /*! \class WFallBCast
        *  \brief Performs broadcast of the last sub-group element to the next sub-group.
        *
        * @tparam SubGroupSize - size of the broadcast blocks.
        */
        template <uint32_t SubGroupSize, uint32_t OpImpl, uint32_t OpCtrl>
        struct WFallBCast : public OpBase<Properties::OP_ID_WFALL_BCAST,
                                          AMDGCN_WAVE_SIZE,
                                          SubGroupSize,
                                          OpImpl,
                                          OpCtrl>
        {
        };

    } // namespace CrossLaneOps

} // namespace rocwmma

#endif // ROCWMMA_CROSS_LANE_OPS_HPP
