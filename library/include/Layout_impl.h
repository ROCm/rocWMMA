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
#ifndef WMMA_LAYOUT_IMPL_H
#define WMMA_LAYOUT_IMPL_H

#include "Layout.h"
#include "MappingUtil.h"
#include "Utils.h"

/**
 * Layout
 */
namespace Layout
{

    /**
     * \ingroup dataLayouts
     * \defgroup ColNT
     * @{
     */

    /**
     *
     * ColNT
     *
     * ColNT signifies that this layout will align contiguous threads
     * in preference of matrix columns. The order that columns map to
     * registers is affected by DataLayout, MaxVectorWidth and BlockDim.
     *
     * DataLayout direction (row major or column major) affects the mapping
     * of contiguous vector neighbours. In row major, vector coordinates
     * are mapped contiguously across columns, and in col major vector
     * coordinates are mapped contiguously across rows.
     *
     *      Matrix Coords (row major)
     *      Vector Dim ------->
     *       V0    V1   V2
     *       _v__ _v__ _v__  ...
     *      |    |    |
     *      |    |    |
     *      | C0 | C1 | C2
     *      |    |    |
     *      |___ |___ |____  ...
     *
     *      V0 = (0, 0)  V1 = (0, 1)  V2 = (0, 2)...
     *
     *     Possible Register file (row major):
     *
     *     Elements 0...............64
     *               ________________
     *      Reg0    |      C0        |
     *      Reg1    |      C1        |
     *      Reg2    |      C2        |
     *      ...     |      ...       |
     *
     *
     *      Matrix Coords (col major):
     *  Vector Dim
     *   |
     *   |        ____ ____ ____  ...
     *   |  V0 > |    |    |
     *   |  V1 > |    |    |
     *   |  V2 > | C0 | C1 | C2
     *   v       |    |    |
     *           |... |... | ...
     *
     *      V0 = (0, 0)  V1 = (1, 0)  V2 = (2, 0)...
     *
     *     Possible Register file (col major):
     *
     *     Elements    0........1.......2......64
     *              ______________________________
     *      Reg0   |  C0E0  |  C1E0  | C2E0 | ... |
     *      Reg1   |  C0E1  |  C1E1  | C2E1 | ... |
     *      Reg2   |  C0E2  |  C1E2  | C2E2 | ... |
     *      ...    |   ...  |  ...   | ...  | ... |
     *
     * Register files provided by ColNT in row major and MaxVectorWidth >= 1
     * are more conducive to MFMA alignment.
     }*/

    /**
     * \ingroup ColNT
     * \defgroup Row Major ColNT
     * @{
     */
    /*
     * Iterative thread offset cycles: Fill MaxVW => Fill BlockDim => Fill K
     *
     * Index on VW segments first, BlockDimSegs second. Below shows the indexing
     * order of columns for two full major cycles:
     *
     * E.g.
     * WaveSize = 64    Iterations = 8
     * BlockDim = 128   BlockK = 8          BlockDimSegs = 2
     * VectorWidth = 2  MaxVectorWidth = 4  VWSegs = 2
     *
     * Minor cycle = VWSegs = 2 iterations
     * Major cycle = VWSegs * BlockDimSegs = 4 iterations
     *
     * iteration offsets:
     * i0 = (0, 0)   i1 = (0, 2)  i2 = (64, 0) i3 = (64, 2)
     * i4 = (0, 4)   i5 = (0, 6)  i6 = (64, 4) i7 = (64, 6)
     *
     *   kDim --------->
     *
     *   i0          i1          i4          i5
     *   v_____ _____v_____ _____v_____ _____v_____ _____
     *   |     |     |     |     |     |     |     |     |
     *   |     |     |     |     |     |     |     |     |
     *   | C0  |  C1 |  C2 |  C3 |  C8 |  C9 | C10 | C11 | ...
     *   |     |     |     |     |     |     |     |     |
     *   |_____|_____|_____|_____|_____|_____|_____|_____|
     *   i2          i3          i6          i7
     *   v_____ _____v_____ _____v_____ _____v_____ _____
     *   |     |     |     |     |     |     |     |     |
     *   |     |     |     |     |     |     |     |     |
     *   | C4  |  C5 |  C6 |  C7 | C12 | C13 | C14 | C15 | ...
     *   |     |     |     |     |     |     |     |     |
     *   |_____|_____|_____|_____|_____|_____|_____|_____|
     *   ^(128, 0)                                       ^(BlockDim, BlockK)
     *   ...                                          ...
     *
     * Register file (for all VectorWidths = [1, MaxVectorWidth]):
     *
     * Elements 0......64
     *          ______
     *  Reg0    |  C0  |
     *  Reg1    |  C1  |
     *  Reg2    |  C2  |
     *  Reg3    |  C3  |
     *  Reg4    |  C4  |
     *  Reg5    |  C5  |
     *  ...       ...
     *  Reg15   |  C15 |
    }*/

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    struct ColNT<BlockDim, BlockK, DataT, row_major, VectorWidth, MaxVectorWidth>
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // Number of threads per wave
                WaveSize = IOTraits::ThreadsPerIO,

                // Number of BlockDim columns gathered per cycle of MaxVW
                MaxKPerIO = WaveSize * MaxVectorWidth / BlockDim,

                // Flag for large BlockDim
                LargeDim = BlockDim >= WaveSize,

                // Number of column segments (> 0 if LargeDim )
                BlockDimSegs = BlockDim / WaveSize,

                // Number of vector width segments
                VWSegs = MaxVectorWidth / VectorWidth,

                // Number of columns per wave (> 0 if !LargeDim)
                WaveSegs = WaveSize / BlockDim,

                // Log2 Values
                Log2BlockDim     = Log2<BlockDim>::value,
                Log2MaxKPerIO    = Log2<MaxKPerIO>::value,
                Log2MaxVW        = Log2<MaxVectorWidth>::value,
                Log2VW           = Log2<VectorWidth>::value,
                Log2WaveSize     = Log2<WaveSize>::value,
                Log2BlockDimSegs = Log2<BlockDimSegs>::value,
                Log2VWSegs       = Log2<VWSegs>::value,
                Log2WaveSegs     = Log2<WaveSegs>::value
            };
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, row_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // TODO: Use constexpr if on C++17
            if(Traits::LargeDim)
            {
                // Base offsetX = threadId.x % ThreadsPerIO
                // Base offsetY = 0
                constexpr int32_t WaveSizeModMask = LsbMask<Traits::Log2WaveSize>::value;
                return std::make_pair(threadIdx.x & WaveSizeModMask, 0);
            }
            else
            {
                // Base offsetX = threadIdx.x % BlockDim
                // Base offsetY = (threadIdx.x / BlockDim) * MaxVectorWidth % Traits::MaxKPerIO;
                constexpr int32_t BlockDimModMask  = LsbMask<Traits::Log2BlockDim>::value;
                constexpr int32_t MaxKPerIOModMask = LsbMask<Traits::Log2MaxKPerIO>::value;

                return std::make_pair(threadIdx.x & BlockDimModMask,
                                      threadIdx.x >> Traits::Log2BlockDim << Traits::Log2MaxVW
                                          & MaxKPerIOModMask);
            }
        }
        __device__ static inline typename Traits::MatrixCoordT incrementalOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if on C++ 17
            if(Traits::LargeDim)
            {
                // incOffsetX:
                // Minor cycle (VWSegs): = (iteration + 1) % VWSegs ? 0 : 64
                // Major cycle (VWSegs * BlockDim):
                // = (iteration + 1) % (VWSegs * BlockDimSegs) ? 0 : -64 * (BlockDimSegs - 1)
                constexpr int32_t IncXMinorStep = Traits::WaveSize;
                constexpr int32_t IncXMajorStep = BlockDim;

                // incOffsetY:
                // Minor cycle (VWSegs): = (iteration + 1) % VWSegs ? VW : -VW * (VWSegs - 1)
                // Major cycle (VWSegs * BlockDim):
                // = (iteration + 1) % (VWSegs * BlockDimSegs) == 0 ? VW : MinorCycle
                constexpr int32_t IncYMinorStep = VectorWidth;
                constexpr int32_t IncYMajorStep = MaxVectorWidth;

                // Bit masking for modulus operation
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t TotalSegsModMask
                    = LsbMask<Traits::Log2VWSegs + Traits::Log2BlockDimSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                int32_t minorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                int32_t majorStepMask = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

                return std::make_pair(
                    (IncXMinorStep & minorStepMask) - (majorStepMask & IncXMajorStep),
                    IncYMinorStep - ((minorStepMask ^ majorStepMask) & IncYMajorStep));
            }
            else
            {
                // incOffsetX: 0
                // incOffsetY:
                // Minor cycle (Every iteration) = VW
                // Major cycle ((iteration + 1) % (VWSegs) == 0) ? (MaxVW * (WaveSegs - 1)) : 0
                constexpr int32_t IncYMinorStep = VectorWidth;
                constexpr int32_t IncYMajorStep = MaxVectorWidth * (Traits::WaveSegs - 1);
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                int32_t majorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                return std::make_pair(0, IncYMinorStep + (majorStepMask & IncYMajorStep));
            }
        }
        __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if on C++17
            if(Traits::LargeDim)
            {
                // Cumulative offsetX = iteration / VWSegs % BlockDimSegs * 64
                // Cumulative offsetY = iteration / TotalSegs * MaxVectorWidth +
                //                      iteration % VWSegs * VectorWidth;
                constexpr int32_t VWSegsModMask       = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t BlockDimSegsModMask = LsbMask<Traits::Log2BlockDimSegs>::value;

                return std::make_pair((iteration & (BlockDimSegsModMask << Traits::Log2VWSegs))
                                          << (Traits::Log2WaveSize - Traits::Log2VWSegs),
                                      (iteration >> (Traits::Log2BlockDimSegs + Traits::Log2VWSegs)
                                                        << Traits::Log2MaxVW)
                                          + ((iteration & VWSegsModMask) << Traits::Log2VW));
            }
            else
            {
                // Cumulative offsetX = 0
                // Cumulative offsetY = iteration / VWSegs * (MaxVW * WaveSegs) + iteration % VWSegs * VW
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                return std::make_pair(
                    0,
                    (iteration >> Traits::Log2VWSegs << (Traits::Log2MaxVW + Traits::Log2WaveSegs))
                        + ((iteration & VWSegsModMask) << Traits::Log2VW));
            }
        }
    };

    /**
     * \ingroup ColNT
     * \defgroup Row Major ColNT
     * @{
     */
    /*
     * Iterative thread offset cycles: Fill MaxVW => Fill BlockDim => Fill K
     *
     * Index on VW segments first, BlockDimSegs second. Below shows the indexing
     * order of columns for two full major cycles:
     *
     * E.g.
     * WaveSize = 64    Iterations = 16
     * BlockDim = 256   BlockK = 4          BlockDimSegs = 2
     * VectorWidth = 1  MaxVectorWidth = 2  VWSegs = 2
     *
     * Minor cycle = VWSegs = 2 iterations
     * Major cycle = VWSegs * BlockDimSegs = 4 iterations
     *
     * iteration offsets:
     * i0  = (0, 0)   i1  = (1, 0)  i2 = (128, 0) i3 = (129, 0)
     * i4  = (0, 1)   i5  = (1, 1)  i6 = (128, 1) i7 = (129, 1)
     * i8  = (0, 2)   i9  = (1, 2)  i10 = (128, 2) i11 = (129, 2)
     * i12 = (0, 3)  i13  = (1, 3)  i14 = (128, 3) i15 = (129, 3)
     *
     *   kDim --------->
     *
     *   i0    i4    i8    i12
     *   v_____v_____v_____v_____
     *   |     |     |     |     |
     *   i1    i5    i9    i13   |
     *   v     v     v     v     |
     *   | C0  | C4  |  C8 | C12 |
     *   |_____|_____|_____|_____|
     *   |     |     |     |     |
     *   |     |     |     |     |
     *   | C1  | C5  |  C9 | C13 |
     *   |     |     |     |     |
     *   i2    i6    i10   i14
     *   v_____v_____v_____v_____
     *   |     |     |     |     |
     *   i3    i7    i11   i15   |
     *   v     v     v     v     |
     *   | C2  | C6  | C10 | C14 |
     *   |     |     |     |     |
     *   |_____|_____|_____|_____|
     *   |     |     |     |     |
     *   |     |     |     |     |
     *   | C3  | C7  | C11 | C15 |
     *   |     |     |     |     |
     *   |_____|_____|_____|_____|
     *   ^(256, 0)               ^(BlockDim, BlockK)
     *
     * Register file:
     *
     * Elements 0...........1.............................................64
     *         ______________________________________________________________
     * Reg0   |  C0E0  |  C0E2 | ... | C0E62  | C1E0  | C1E2  | ... |  C1E62 |  (MaxVW elements 0 of C0, C1)
     * Reg1   |  C0E1  |  C0E3 | ... | C0E63  | C1E1  | C1E3  | ... |  C1E63 |  (MaxVW elements 1 of C0, C1)
     * Reg2   |  C2E0  |  C2E2 | ... | C2E62  | C3E0  | C3E2  | ... |  C3E62 |  (MaxVW elements 0 of C2, C3)
     * Reg3   |  C2E1  |  C2E3 | ... | C2E63  | C3E1  | C3E3  | ... |  C3E63 |  (MaxVW elements 1 of C2, C3)
     * Reg4   |  C4E0  |  C4E2 | ... | C4E62  | C5E0  | C5E2  | ... |  C5E62 |  (MaxVW elements 0 of C4, C5)
     * Reg5   |  C4E1  |  C4E3 | ... | C4E63  | C5E1  | C5E3  | ... |  C5E63 |  (MaxVW elements 1 of C4, C5)
     * ...      ...
     * Reg15  |  C14E1 | C14E3 | ... | C14E63 | C15E1 | C15E3 | ... | C15E63 |  (MaxVW elements 1 of C14, C15)
    */

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    struct ColNT<BlockDim, BlockK, DataT, col_major, VectorWidth, MaxVectorWidth>
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // Number of threads per wave
                WaveSize = IOTraits::ThreadsPerIO,

                // Number of elements per IO of MaxVW
                MaxElementsPerIO = WaveSize * MaxVectorWidth,

                // Number of BlockDim columns gathered per cycle of MaxVW
                MaxKPerIO = MaxElementsPerIO / BlockDim,

                // Flag for large BlockDim
                LargeDim = BlockDim >= MaxElementsPerIO,

                // Number of column segments (> 0 if LargeDim )
                BlockDimSegs = BlockDim / MaxElementsPerIO,

                // Number of vector width segments
                VWSegs = MaxVectorWidth / VectorWidth,

                // Log2 Values
                Log2BlockDim         = Log2<BlockDim>::value,
                Log2MaxElementsPerIO = Log2<MaxElementsPerIO>::value,
                Log2MaxKPerIO        = Log2<MaxKPerIO>::value,
                Log2MaxVW            = Log2<MaxVectorWidth>::value,
                Log2VW               = Log2<VectorWidth>::value,
                Log2WaveSize         = Log2<WaveSize>::value,
                Log2BlockDimSegs     = Log2<BlockDimSegs>::value,
                Log2VWSegs           = Log2<VWSegs>::value,
            };
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, col_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;

            static_assert(BlockK >= MaxVectorWidth, "BlockK must be at least MaxVectorWidth");
            static_assert(BlockK % MaxVectorWidth == 0,
                          "BlockK must be a multiple of MaxVectorWidth");
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Base offsetX = threadId.x * MaxVW % MaxElementsPerIO
                // Base offsetY = 0
                constexpr int32_t MaxElementsPerIOModMask
                    = LsbMask<Traits::Log2MaxElementsPerIO>::value;

                return std::make_pair(threadIdx.x << Traits::Log2MaxVW & MaxElementsPerIOModMask,
                                      0);
            }
            else
            {
                // Base offsetX = threadId.x * MaxVW % BlockDim
                // Base offsetY = threadId.x * MaxVW / BlockDim % MaxKPerIO
                constexpr int32_t BlockDimModMask  = LsbMask<Traits::Log2BlockDim>::value;
                constexpr int32_t MaxKPerIOModMask = LsbMask<Traits::Log2MaxKPerIO>::value;

                return std::make_pair(threadIdx.x << Traits::Log2MaxVW & BlockDimModMask,
                                      threadIdx.x << Traits::Log2MaxVW >> Traits::Log2BlockDim
                                          & MaxKPerIOModMask);
            }
        }

        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT incrementalOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                constexpr int32_t IncX0MinorStep = VectorWidth;
                constexpr int32_t IncX0MajorStep = MaxVectorWidth;

                constexpr int32_t IncX1MinorStep = Traits::MaxElementsPerIO;
                constexpr int32_t IncX1MajorStep = BlockDim;

                constexpr int32_t IncYMinorStep = 0;
                constexpr int32_t IncYMajorStep = 1;

                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t TotalSegsModMask
                    = LsbMask<Traits::Log2BlockDimSegs + Traits::Log2VWSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                int32_t VWSegsStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                int32_t TotalSegsStepMask
                    = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

                return std::make_pair(IncX0MinorStep - (VWSegsStepMask & IncX0MajorStep)
                                          + (VWSegsStepMask & IncX1MinorStep)
                                          - (TotalSegsStepMask & IncX1MajorStep),
                                      TotalSegsStepMask & IncYMajorStep);
            }
            else
            {
                constexpr int32_t IncXMinorStep = VectorWidth;
                constexpr int32_t IncXMajorStep = MaxVectorWidth;
                constexpr int32_t IncYMinorStep = 0;
                constexpr int32_t IncYMajorStep = Traits::MaxKPerIO;
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                int32_t majorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                // Reference calculation:
                // Iterative offsetX = VW - ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxVW
                // Iterative offsetY = ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxKPerIO
                return std::make_pair(IncXMinorStep - (majorStepMask & IncXMajorStep),
                                      majorStepMask & IncYMajorStep);
            }
        }

        // Cumulative iteration offset
        __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                constexpr int32_t VWSegsModMask       = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t BlockDimSegsModMask = LsbMask<Traits::Log2BlockDimSegs>::value;

                // Cumulative offsetX = (iteration / VWSegs) % BlockDimSegs * MaxElementsPerIO +
                //                      (iteration % VWSegs) * VW,
                // Cumulative offsetY = iteration / TotalSegs;
                return std::make_pair(
                    (iteration << (Traits::Log2MaxElementsPerIO - Traits::Log2VWSegs))
                        & (BlockDimSegsModMask << Traits::Log2MaxElementsPerIO)
                                  + (iteration & VWSegsModMask)
                              << Traits::Log2VW,
                    iteration >> (Traits::Log2VWSegs + Traits::Log2BlockDimSegs));
            }
            else
            {
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                // Cumulative offsetX = (iteration % VWSegs) * VW
                // Cumulative offsetY = iteration / VWSegs * (MaxKPerIO)
                return std::make_pair((iteration & VWSegsModMask) << Traits::Log2VW,
                                      iteration >> Traits::Log2VWSegs << Traits::Log2MaxKPerIO);
            }
        }
    };

    ///////////////////////////////////////////////////////////
    ////////////////////////  RowNT  //////////////////////////
    ///////////////////////////////////////////////////////////

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    __device__ inline auto
        RowNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>::baseOffset() ->
        typename Traits::MatrixCoordT
    {
        return std::swap(Traits::OrthoLayout::baseOffset());
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    __device__ inline auto
        RowNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>::incrementalOffset(
            uint32_t iteration) -> typename Traits::MatrixCoordT
    {
        return std::swap(Traits::OrthoLayout::incrementalOffset(iteration));
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    __device__ inline auto
        RowNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>::cumulativeOffset(
            uint32_t iteration) -> typename Traits::MatrixCoordT
    {
        return std::swap(Traits::OrthoLayout::cumulativeOffset(iteration));
    }

} // namespace Layout

#endif // WMMA_LAYOUT_IMPL_H
