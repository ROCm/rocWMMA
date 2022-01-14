/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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
#ifndef WMMA_LAYOUT_H
#define WMMA_LAYOUT_H

#include <hip/hip_runtime.h>

#include "Types.h"
#include <tuple>

// FWD decl.
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth>
struct amdgcn_io_traits;

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
struct MappingUtil;

/**
 * \ingroup wmma
 * \defgroup dataLayouts
 *
 * @brief Definition and metadata on supported data layout of matrices.
 *
 * These layouts are based in Matrix Space. They are to map each of the wave lanes
 * into corresponding row / col coordinates for a particular memory layout.
 * For example, the A matrix loads columns of size BlockDim in the K direction.
 * The B matrix loads rows of size BlockDim in the K direction.
 *
 * Each of these layouts is indexed differently, especially when different datatypes
 * and load widths are used. These classes are intended to address these matrix
 * space indexing challenges.
 */

/**
 * Layout
 */
namespace Layout
{
    /**
 * \ingroup dataLayouts
 * @{
 */
    /**
 *
 * *Matrix layout:*
 *
 * BlockDim = column size
 *
 * BlockK = column count
 *
 *
 *      kDim ->
 *      (0, 0)              (0, BlockK - 1)
 *      v______________  ... v____
 *      |    |    |          |    |
 *      |    |    |          |    |
 *      | C0 | C1 | C2       | Ck |
 *      |    |    |          |    |
 *      |___ |___ |____  ... |____|
 *      ^(BlockDim - 1, 0)   ^(BlockDim - 1, BlockK - 1)
 *
 * *Register layout:*
 *
 * KPerIO == # of columns per IO (either load or store)
 *
 * E.g. BlockDim = 32, VW = 1, KPerIO = 2, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  C0   |   C1  |
 *       ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 4, KPerIO = 8, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *                _______________
 *       Reg0    |  C0   |   C4  |
 *       Reg1    |  C1   |   C5  |
 *       Reg2    |  C2   |   C6  |
 *       Reg3    |  C3   |   C7  |
 *        ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 1, KPerIO = 2, DataT = f32, DataLayout = col_major
 *
 *       Elements 0.....31 32.....64
 *                _______________
 *       Reg0    |  C0   |   C1  |
 *        ...       ...      ...
 *
 *
 *  E.g. BlockDim = 32, VW = 4, KPerIO = 8, DataT = f32, DataLayout = col_major
 *
 *       Elements 0.............1    ...    7...........8..........9....  ...     64
 *               ___________________    _______________________________   ...  _________
 *       Reg0    |  C0E0   |   C0E4  ...   C0E28   |   C1E0   |   C1E4  |  ... |  C8E28  |
 *       Reg1    |  C0E1   |   C0E5  ...   C0E29   |   C1E1   |   C1E5  |  ... |  C8E29  |
 *       Reg2    |  C0E2   |   C0E6  ...   C0E30   |   C1E2   |   C1E6  |  ... |  C8E30  |
 *       Reg3    |  C0E3   |   C0E7  ...   C0E31   |   C1E3   |   C1E7  |  ... |  C8E31  |
 *        ...       ...      ...
 *
 *
 */
    ////////////// Col /////////////////////////
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth>
    struct Col;

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth>
    struct Col<BlockDim, BlockK, DataT, row_major, VectorWidth>
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // Number of BlockDim strides per I/O operation
                KPerIO = ceilDiv((uint32_t)IOTraits::ElementsPerIO, BlockDim),

                // Minimum IOCount to implement this layout.
                MinIOCount = ceilDiv(BlockDim, (uint32_t)IOTraits::ThreadsPerIO),

                // Addressing flag for larger BlockDim than gather size
                LargeDim = BlockDim >= (uint32_t)IOTraits::ThreadsPerIO
            };

            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, row_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        // Initial thread offset alignment
        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation:
                // Base offsetX = threadId.x % ThreadsPerIO
                // Base offsetY = 0
                constexpr int32_t Log2WaveSize = Log2<IOTraits::ThreadsPerIO>::value;
                constexpr int32_t WaveSizeModMask = LsbMask<Log2WaveSize>::value;
                
                return std::make_pair(threadIdx.x & WaveSizeModMask, 0);
            }
            else
            {
                // Reference calculation:
                // Base offsetX = threadId.x % BlockDim
                // Base offsetY = threadId.x / BlockDim * VW % KPerIO
                constexpr int32_t Log2BlockDim = Log2<BlockDim>::value;
                constexpr int32_t Log2VW = Log2<VectorWidth>::value;
                constexpr int32_t Log2KPerIO = Log2<Traits::KPerIO>::value;
                constexpr int32_t BlockDimModMask = LsbMask<Log2BlockDim>::value;
                constexpr int32_t VWShiftMask = LsbMask<Log2VW>::value;
                constexpr int32_t KPerIOModMask = LsbMask<Log2KPerIO>::value;
                
                // Must mask out the (Log2VW bits) because VW shift is AFTER the division
                return std::make_pair(threadIdx.x & BlockDimModMask,
                                     (threadIdx.x >> (Log2BlockDim - Log2VW)) & (VWShiftMask ^ KPerIOModMask));
            }
        }

        // Cumulative iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation:
                // Cumulative offsetX = iteration % BlockDimSegs * ThreadsPerIO
                // Cumulative offsetY = iteration / BlockDimSegs * VW
                constexpr int32_t BlockDimSegs = BlockDim / IOTraits::ThreadsPerIO;
                constexpr int32_t Log2BlockDimSegs = Log2<BlockDimSegs>::value;
                constexpr int32_t Log2ThreadsPerIO = Log2<IOTraits::ThreadsPerIO>::value;
                constexpr int32_t Log2VW = Log2<VectorWidth>::value;
                constexpr int32_t BlockDimSegsModMask = LsbMask<Log2BlockDimSegs>::value;

                return std::make_pair((iteration & BlockDimSegsModMask) << Log2ThreadsPerIO,
                                    (iteration >> Log2BlockDimSegs) << Log2VW);
            }
            else
            {
                // Reference calculation:
                // Cumulative offsetX = 0
                // Cumulative offsetY = i * KPerIO
                constexpr int32_t Log2KPerIO = Log2<Traits::KPerIO>::value;
                return std::make_pair((uint32_t)0, iteration << Log2KPerIO); // Shift K
            }
        }

        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationIncOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation: increments cycle every multiple of BlockDimSegs.
                // Incremental offsetX: (iteration + 1) % BlockDimSegs ?
                //                      move back to initial BlockDim segment:
                //                      move to next BlockDim segment
                // Incremental offsetY: (iteration + 1) % BlockDimSegs ?
                //                      move VW columns over :
                //                      stay in current column 
                constexpr int32_t BlockDimSegs = BlockDim / IOTraits::ThreadsPerIO;
                constexpr int32_t IncXMinorStep = IOTraits::ThreadsPerIO;
                constexpr int32_t IncXMajorStep = BlockDim;
                constexpr int32_t IncYMinorStep = 0;
                constexpr int32_t IncYMajorStep = VectorWidth;

                constexpr int32_t Log2BlockDimSegs = Log2<BlockDimSegs>::value;
                constexpr int32_t ModBlockDimSegsMask = LsbMask<Log2BlockDimSegs>::value;

                // Any remainder bits detected, majorStep = 0x0
                // No remainder bits detected, majorStep = 0xFFFFFFFF
                auto majorStepMask = static_cast<bool>((iteration + 1) & ModBlockDimSegsMask) - 1;

                return std::make_pair(
                    IncXMinorStep - (majorStepMask & IncXMajorStep),
                    majorStepMask & IncYMajorStep);
            }
            else
            {
                return std::make_pair((uint32_t)0, IOTraits::KPerIO); // Shift K
            }
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK);
        }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth>
    struct Col<BlockDim, BlockK, DataT, col_major, VectorWidth>
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // Number of BlockDim strides per I/O operation
                KPerIO = ceilDiv((uint32_t)IOTraits::ElementsPerIO, BlockDim),

                // Minimum IOCount to implement this layout.
                MinIOCount = ceilDiv(BlockDim, (uint32_t)IOTraits::ThreadsPerIO),

                // Addressing flag for larger BlockDim than gather size
                LargeDim = BlockDim >= (uint32_t)IOTraits::ElementsPerIO
            };

            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, row_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        // Initial thread offset alignment
        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation:
                // Base offsetX = threadId.x * VW % ElementsPerIO
                // Base offsetY = threadId.x * VW / BlockDim % KPerIO
                constexpr int32_t Log2ElementsPerIO = Log2<IOTraits::ElementsPerIO>::value;
                constexpr int32_t Log2BlockDim = Log2<BlockDim>::value;
                constexpr int32_t Log2VW = Log2<VectorWidth>::value;
                constexpr int32_t Log2KPerIO = Log2<Traits::KPerIO>::value;
                constexpr int32_t ElementsPerIOModMask = LsbMask<Log2ElementsPerIO>::value;
                constexpr int32_t KPerIOModMask = LsbMask<Log2KPerIO>::value;
                
                // Keep the (-Log2VW) bits because VW shift is BEFORE division
                return std::make_pair((threadIdx.x << Log2VW) & ElementsPerIOModMask,
                                      (threadIdx.x >> (Log2BlockDim - Log2VW)) & KPerIOModMask);
            }
            else
            {
                // Reference calculation:
                // Base offsetX = threadId.x * VW % BlockDim
                // Base offsetY = threadId.x * VW / BlockDim % KPerIO
                constexpr int32_t Log2BlockDim = Log2<BlockDim>::value;
                constexpr int32_t Log2VW = Log2<VectorWidth>::value;
                constexpr int32_t Log2KPerIO = Log2<Traits::KPerIO>::value;
                constexpr int32_t BlockDimModMask = LsbMask<Log2BlockDim>::value;
                constexpr int32_t KPerIOModMask = LsbMask<Log2KPerIO>::value;
                
                // Keep the (-Log2VW) bits because VW shift is BEFORE division
                return std::make_pair((threadIdx.x << Log2VW) & BlockDimModMask,
                                      (threadIdx.x >> (Log2BlockDim - Log2VW)) & KPerIOModMask);
            }
        }

        // Cumulative iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation:
                // Cumulative offsetX = iteration % BlockDimSegs * ElementsPerIO
                // Cumulative offsetY = iteration / BlockDimSegs
                constexpr int32_t BlockDimSegs = BlockDim / IOTraits::ElementsPerIO;
                constexpr int32_t Log2BlockDimSegs = Log2<BlockDimSegs>::value;
                constexpr int32_t Log2ElementsPerIO = Log2<IOTraits::ElementsPerIO>::value;
                constexpr int32_t Log2VW = Log2<VectorWidth>::value;
                constexpr int32_t BlockDimSegsModMask = LsbMask<Log2BlockDimSegs>::value;

                return std::make_pair((iteration & BlockDimSegsModMask) << Log2ElementsPerIO,
                                      iteration >> Log2BlockDimSegs);
            }
            else
            {
                // Reference calculation:
                // Cumulative offsetX = 0
                // Cumulative offsetY = i * KPerIO
                constexpr int32_t Log2KPerIO = Log2<Traits::KPerIO>::value;
                return std::make_pair((uint32_t)0, iteration << Log2KPerIO); // Shift K
            }
        }

        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationIncOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Reference calculation: increments cycle every multiple of BlockDimSegs.
                // Incremental offsetX: (iteration + 1) % BlockDimSegs ?
                //                      move back to initial BlockDim segment:
                //                      move to next BlockDim segment
                // Incremental offsetY: (iteration + 1) % BlockDimSegs ?
                //                      move 1 column over :
                //                      stay in current column

                // Note: This function may be called in hot loops, so attempt to avoid branching
                // by computing both cases and masking out the values that are not needed. 
                constexpr int32_t BlockDimSegs = BlockDim / IOTraits::ElementsPerIO;
                constexpr int32_t IncXMinorStep = IOTraits::ElementsPerIO;
                constexpr int32_t IncXMajorStep = IncXMinorStep << Log2BlockDimSegs;
                constexpr int32_t IncYMinorStep = 0;
                constexpr int32_t IncYMajorStep = 1;

                constexpr int32_t Log2BlockDimSegs = Log2<BlockDimSegs>::value;
                constexpr int32_t ModBlockDimSegsMask = LsbMask<Log2BlockDimSegs>::value;

                // Any remainder bits detected, majorStep = 0x0
                // No remainder bits detected, majorStep = 0xFFFFFFFF
                auto majorStepMask = static_cast<bool>((iteration + 1) & ModBlockDimSegsMask) - 1;

                // Use copy mask to override major / minor steps without branching
                return std::make_pair(
                    IncXMinorStep - (majorStepMask & IncXMajorStep),
                    majorStepMask & IncYMajorStep);
            }
            else
            {
                // Reference calculation:
                // Cumulative offsetX = 0
                // Cumulative offsetY = KPerIO
                return std::make_pair((uint32_t)0, IOTraits::KPerIO); // Shift K
            }
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK);
        }
    };

    /**
 * \ingroup dataLayouts
 * @{
 */
    /**
 *
 * *Matrix layout:*
 *
 * Common usage: Matrix B, C, D
 *
 * BlockDim = row size
 *
 * BlockK = row count
 *
 *      BlockDim ->
 *      (0, 0)                 (0, BlockDim - 1)
 *      v______________  ...  _v__
 *      |__________R0__  ...  ____|
 *      |__________R1__  ...  ____|
 *      |__________R2__  ...  ____|
 *      |          ...   ...      |
 *      |__________Rk__  ...  ____|
 *      ^(BlockK - 1, 0)       ^(BlockK - 1, BlockDim - 1)
 *
 * *Register layout:*
 *
 * ElementsPerThread == VectorWidth
 *
 * KPerIO == # of rows per IO (either load or store)
 *
 * E.g. BlockDim = 32, VW = 1, KPerIO = 2, DataT = f32, DataLayout = col_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  R0   |   R1  |
 *      ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 4, KPerIO = 8, DataT = f32, DataLayout = col_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  R0   |   R4  |
 *      Reg1    |  R1   |   R5  |
 *      Reg2    |  R2   |   R6  |
 *      Reg3    |  R3   |   R7  |
 *       ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 1, KPerIO = 2, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  R0   |   R1
 *      ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 4, KPerIO = 8, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.............1    ...    7...........8..........9....   ...     64
 *               ___________________    _______________________________   ...  _________
 *      Reg0    |  R0E0   |   R0E4  ...   R0E28   |   R1E0   |   R1E4  |  ... |  R8E28  |
 *      Reg1    |  R0E1   |   R0E5  ...   R0E29   |   R1E1   |   R1E5  |  ... |  R8E29  |
 *      Reg2    |  R0E2   |   R0E6  ...   R0E30   |   R1E2   |   R1E6  |  ... |  R8E30  |
 *      Reg3    |  R0E3   |   R0E7  ...   R0E31   |   R1E3   |   R1E7  |  ... |  R8E31  |
 *       ...       ...      ...
 *
 */
    ////////////// Row /////////////////////////
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth>
    struct Row
    {
        struct Traits
        {
            // Orthogonal layout has mirrored coordinates
            // and
            using OrthoLayout = Col<BlockDim,
                                    BlockK,
                                    DataT,
                                    std::conditional_t<std::is_same<DataLayout, row_major>::value,
                                                       col_major,
                                                       row_major>,
                                    VectorWidth>;
            enum : uint32_t
            {
                //MinK       = OrthoLayout::Traits::MinK,
                MinIOCount = OrthoLayout::Traits::MinIOCount
            };

            using MappingUtil  = MappingUtil<BlockK, BlockDim, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            return std::swap(Traits::OrthoLayout::baseOffset());
        }

        // Cumulative iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationOffset(uint32_t iteration)
        {
            return std::swap(Traits::OrthoLayout::iterationOffset(iteration));
        }

        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT iterationIncOffset()
        {
            return std::swap(Traits::OrthoLayout::iterationIncOffset());
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::swap(Traits::OrthoLayout::blockOffset());
        }
    };

    /**
 * \ingroup dataLayouts
 * @{
 */
    /**
 * *Matrix layout:*
 *
 * BlockDim = column size
 *
 * BlockK = column count
 *
 * N = Max vector width
 *
 * VW = Actual vector width
 *
 *      kDim ->
 *      (0, 0)              (0, BlockK - 1)
 *      v______________  ... v____
 *      |    |    |          |    |
 *      |    |    |          |    |
 *      | C0 | C1 | C2       | Ck |
 *      |    |    |          |    |
 *      |___ |___ |____  ... |____|
 *      ^(BlockDim - 1, 0)   ^(BlockDim - 1, BlockK - 1)
 *
 * *Register layout:*
 *
 * *Guarantees:*
 *
 * ColNT guarantees the following register format, regardless of VW and data layout.
 *
 * Register 0 to contain Cols (i % N) == 0
 *
 * Register 1 to contain Cols (i % N) == 1
 *
 * Register 2 to contain Cols (i % N) == 2
 *
 * ...
 *
 * *Limitations:*
 *
 * col_major data format is not supported for VW > 1, as it produces
 * incorrect mapping
 *
 * KPerIO == # of columns per IO (either load or store)
 *
 *
 * E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  C0   |   C4  |
 *       ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 4, N = 4, KPerIO = 8, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  C0   |   C4  |
 *      Reg1    |  C1   |   C5  |
 *      Reg2    |  C2   |   C6  |
 *      Reg3    |  C3   |   C7  |
 *       ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = col_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  C0   |   C4  |
 *       ...       ...      ...
 *
 * @note
 * E.g. BlockDim = 32, VW = 4, N = 4, KPerIO = 8, DataT = f32, DataLayout = col_major
 * Is NOT implemented due to incorrect mapping with col_major and VW = 4
 */
    ////////////// ColNT /////////////////////////

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
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
                WaveSize = IOTraits::ThreadsPerIO

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
                Log2BlockDim = Log2<BlockDim>::value,
                Log2MaxKPerIO = Log2<MaxKPerIO>::value,
                Log2MaxVW = Log2<MaxVectorWidth>::value,
                Log2VW = Log2<VectorWidth>::value,
                Log2WaveSize = Log2<WaveSize>::value,
                Log2BlockDimSegs = Log2<BlockDimSegs>::value,
                Log2VWSegs = Log2<VWSegs>::value,
                Log2WaveSegs = Log2<WaveSegs>::value
            };
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
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
                constexpr int32_t BlockDimModMask = LsbMask<Traits::Log2BlockDim>::value;
                constexpr int32_t MaxVWShiftMask = LsbMask<Traits::Log2MaxVW>::value;
                constexpr int32_t MaxKPerIOModMask = LsbMask<Traits::Log2MaxKPerIO>::value;
                
                // Mask out the Log2MaxVW bits because shift left is AFTER the division
                return std::make_pair(threadIdx.x & BlockDimModMask,
                                     (threadIdx.x >> (Traits::Log2BlockDim - Traits::Log2MaxVW)) & (MaxVWShiftMask ^ MaxKPerIOModMask));
            }
        }
        
        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT incrementalOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if on C++ 17
            if(Traits::LargeDim)
            {
                /*
                2D Coord iteration:
                Minor indexing on VWSegs
                Major indexing on BlockDimSegs * VWSegs
                
                Index on VW segments first, BlockDimSegs second. Below shows the indexing
                order of columns for two full major cycles:
                
                E.g. 
                WaveSize = 64    Iterations = 8 
                BlockDim = 128   BlockK = 8          BlockDimSegs = 2
                VectorWidth = 2  MaxVectorWidth = 4  VWSegs = 2
                              
                Minor cycle = VWSegs = 2 iterations
                Major cycle = VWSegs * BlockDimSegs = 4 iterations

                iterations:
                i0 = (0, 0)   i1 = (0, 2)  i2 = (64, 0) i3 = (64, 2)
                i4 = (0, 4)   i5 = (0, 6)  i6 = (64, 4) i7 = (64, 6)
                                
                    kDim --------->
                
                    i0          i1          i4          i5
                    v_____ _____v_____ _____v_____ _____v_____ _____
                    |     |     |     |     |     |     |     |     |
                    |     |     |     |     |     |     |     |     |
                    | C0  |  C1 |  C2 |  C3 |  C8 |  C9 | C10 | C11 | ...
                    |     |     |     |     |     |     |     |     |
                    |_____|_____|_____|_____|_____|_____|_____|_____|
                    i2          i3          i6          i7
                    v_____ _____v_____ _____v_____ _____v_____ _____
                    |     |     |     |     |     |     |     |     |
                    |     |     |     |     |     |     |     |     |
                    | C4  |  C5 |  C6 |  C7 | C12 | C13 | C14 | C15 | ...
                    |     |     |     |     |     |     |     |     |
                    |_____|_____|_____|_____|_____|_____|_____|_____|
                    ^(128, 0)                                       ^(BlockDim, BlockK)
                    ...                                          ...

                Register file:

                Elements 0......64
                         ______
                Reg0    |  C0  |
                Reg1    |  C1  |
                Reg2    |  C2  |
                Reg3    |  C3  |
                Reg4    |  C4  |
                Reg5    |  C5  |
                ...       ...   
                Reg15   |  C15 |
                */

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
                constexpr int32_t TotalSegsModMask = LsbMask<Traits::Log2VWSegs + Traits::Log2BlockDimSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                auto minorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                auto majorStepMask = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

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
                constexpr int32_t IncYMajorStep = MaxVW * (Traits::WaveSegs - 1);
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                
                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                auto int32_t majorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                return std::make_pair(0, IncYMinorStep + majorStepMask & IncYMajorStep);
            }

            __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration)
            {
                // TODO: Use constexpr if on C++17
                if(Traits::LargeDim)
                {
                    // Cumulative offsetX = iteration / VWSegs % BlockDimSegs * 64
                    // Cumulative offsetY = iteration / TotalSegs * MaxVectorWidth + iteration % VWSegs * VectorWidth;
                    constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                    constexpr int32_t BlockDimSegsModMask = LsbMask<Traits::Log2BlockDimSegs>::value;
                    
                    return std::make_pair(iteration & (BlockDimSegsModMask << Traits::Log2VWSegs) << (Traits::Log2WaveSize - Traits::Log2VWSegs),
                                        (iteration >> (Traits::Log2BlockDimSegs + Traits::Log2VWSegs) << Traits::Log2MaxVW) + (iteration & VWSegsModMask << Traits::Log2VW));
                }
                else
                {
                    // Cumulative offsetX = 0
                    // Cumulative offsetY = iteration / VWSegs * (MaxVW * WaveSegs) + iteration % VWSegs * VW
                    constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                    return std::make_pair(0, (iteration >> Traits::Log2VWSegs << (Traits::Log2MaxVW + Traits::Log2WaveSegs)) + (iteration & VWSegsModMask << Traits::Log2VW));
                }
            }
            /*
            E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = row_major

            Elements 0.....31 32.....64
                    _______________
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  C3   |   C7  |
            ...       ...      ...

            The startCoord points threads 0-31 at elements of C0, threads 32-63 at elements of C4.

            If not a full load, there are two increment cycles, one major step size and one minor step size.
            The minor cycle increments every iteration.
            The major cycle increments on every iteration that has loaded 4 registers.

            major = 4
            minor = 1

            i = 0:
            Reg0    |  C0   |   C4  |
            Reg1    |  --   |   --  |
            Reg2    |  --   |   --  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 1:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  --   |   --  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 2:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 3:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  C3   |   C7  |
            inc: 1 * majorStep + minor = 5 Col
            */
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK); // Next block
        }
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
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
                Log2BlockDim = Log2<BlockDim>::value,
                Log2MaxElementsPerIO = Log2<MaxElementsPerIO>::value,
                Log2MaxKPerIO = Log2<MaxKPerIO>::value,
                Log2MaxVW = Log2<MaxVectorWidth>::value,
                Log2VW = Log2<VectorWidth>::value,
                Log2WaveSize = Log2<WaveSize>::value,
                Log2BlockDimSegs = Log2<BlockDimSegs>::value,
                Log2VWSegs = Log2<VWSegs>::value,
            };
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                // Base offsetX = threadId.x * MaxVW % MaxElementsPerIO
                // Base offsetY = 0
                constexpr int32_t MaxElementsPerIOModMask = LsbMask<Log2MaxElementsPerIO>::value;

                return std::make_pair((threadIdx.x << Traits::Log2MaxVW) & MaxElementsPerIOModMask,
                                      0);
            }
            else
            {
                // Base offsetX = threadId.x * MaxVW % BlockDim
                // Base offsetY = threadId.x * MaxVW / BlockDim % MaxKPerIO
                constexpr int32_t BlockDimModMask = LsbMask<Log2BlockDim>::value;
                constexpr int32_t MaxKPerIOModMask = LsbMask<Log2MaxKPerIO>::value;
                
                // Keep the (-Log2VW) bits because VW shift is BEFORE division
                return std::make_pair((threadIdx.x << Traits::Log2MaxVW) & BlockDimModMask,
                                      (threadIdx.x >> (Traits::Log2BlockDim - Traits::Log2MaxVW)) & MaxKPerIOModMask);
            }
        }

        // Incremental iteration offset
        __device__ static inline typename Traits::MatrixCoordT incrementalOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                /*
                2D iteration:
                Minor indexing on VWSegs
                Major indexing on BlockDimSegs * VWSegs
                
                Index BlockDimSegs second. Below shows the indexing
                order of columns for two full major cycles (8 iterations -> BlockK = 8):
                
                E.g. 
                WaveSize = 64    Iterations = 16 
                BlockDim = 256   BlockK = 4          BlockDimSegs = 2
                VectorWidth = 1  MaxVectorWidth = 2  VWSegs = 2
                              
                Minor cycle = VWSegs = 2 iterations
                Major cycle = VWSegs * BlockDimSegs = 4 iterations

                iterations :
                i0  = (0, 0)   i1  = (1, 0)  i2 = (128, 0) i3 = (129, 0)
                i4  = (0, 1)   i5  = (1, 1)  i6 = (128, 1) i7 = (129, 1)
                i8  = (0, 2)   i9  = (1, 2)  i10 = (128, 2) i11 = (129, 2)
                i12 = (0, 3)  i13  = (1, 3)  i14 = (128, 3) i15 = (129, 3)
                                
                    kDim --------->

                    i0    i4    i8    i12
                    v_____v_____v_____v_____
                    |     |     |     |     |
                    i1    i5    i9    i13   |
                    v     v     v     v     |
                    | C0  | C4  |  C8 | C12 |
                    |_____|_____|_____|_____|
                    |     |     |     |     |
                    |     |     |     |     |
                    | C1  | C5  |  C9 | C13 |
                    |     |     |     |     |
                    i2    i6    i10   i14
                    v_____v_____v_____v_____
                    |     |     |     |     |
                    i3    i7    i11   i15   |
                    v     v     v     v     |
                    | C2  | C6  | C10 | C14 |
                    |     |     |     |     |
                    |_____|_____|_____|_____|
                    |     |     |     |     |
                    |     |     |     |     |
                    | C3  | C7  | C11 | C15 |
                    |     |     |     |     |
                    |_____|_____|_____|_____|
                    ^(256, 0)               ^(BlockDim, BlockK)

                Register file:

                Elements 0......................................................64
                        ____________________________________________________________
                Reg0   |  C0E0  |  C0E2 | ... | C0E62  | C1E0  | C1E2  | ... |  C1E62 |  (MaxVW elements 0 of C0, C1)
                Reg1   |  C0E1  |  C0E3 | ... | C0E63  | C1E1  | C1E3  | ... |  C1E63 |  (MaxVW elements 1 of C0, C1)
                Reg2   |  C2E0  |  C2E2 | ... | C2E62  | C3E0  | C3E2  | ... |  C3E62 |  (MaxVW elements 0 of C2, C3)
                Reg3   |  C2E1  |  C2E3 | ... | C2E63  | C3E1  | C3E3  | ... |  C3E63 |  (MaxVW elements 1 of C2, C3)
                Reg4   |  C4E0  |  C4E2 | ... | C4E62  | C5E0  | C5E2  | ... |  C5E62 |  (MaxVW elements 0 of C4, C5)
                Reg5   |  C4E1  |  C4E3 | ... | C4E63  | C5E1  | C5E3  | ... |  C5E63 |  (MaxVW elements 1 of C4, C5)
                ...      ...   
                Reg15  |  C14E1 | C14E3 | ... | C14E63 | C15E1 | C15E3 | ... | C15E63 |  (MaxVW elements 1 of C14, C15)
                */

                constexpr int32_t IncX0MinorStep = VectorWidth;
                constexpr int32_t IncX0MajorStep = MaxVectorWidth;
                
                constexpr int32_t IncX1MinorStep = Traits::MaxElementsPerIO;
                constexpr int32_t IncX1MajorStep = BlockDim;

                constexpr int32_t IncYMinorStep = 0;
                constexpr int32_t IncYMajorStep = 1;

                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t TotalSegsModMask = LsbMask<Traits::Log2BlockDimSegs + Traits::Log2VWSegs>::value;

                // Any remainder bits detected, mask = 0x0
                // No remainder bits detected, mask = 0xFFFFFFFF
                auto int32_t VWSegsStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                auto int32_t TotalSegsStepMask = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

                return std::make_pair(
                    IncX0MinorStep - (VWSegsStepMask & IncX0MajorStep) + (VWSegsStepMask & IncX1MinorStep) - (TotalSegsStepMask & IncX1MajorStep),
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
                auto int32_t majorStepMask = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                // Reference calculation:
                // Iterative offsetX = VW - ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxVW
                // Iterative offsetY = ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxKPerIO
                return std::make_pair(IncXMinorStep - (majorStepMask & IncXMajorStep),
                                      majorStepMask & IncYMajorStep);
            }
            /*
            E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = row_major

            Elements 0.....31 32.....64
                    _______________
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  C3   |   C7  |
            ...       ...      ...

            The startCoord points threads 0-31 at elements of C0, threads 32-63 at elements of C4.

            If not a full load, there are two increment cycles, one major step size and one minor step size.
            The minor cycle increments every iteration.
            The major cycle increments on every iteration that has loaded 4 registers.

            major = 4
            minor = 1

            i = 0:
            Reg0    |  C0   |   C4  |
            Reg1    |  --   |   --  |
            Reg2    |  --   |   --  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 1:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  --   |   --  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 2:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  --   |   --  |
            inc: 0 * majorStep + minor = 1 Col

            i = 3:
            Reg0    |  C0   |   C4  |
            Reg1    |  C1   |   C5  |
            Reg2    |  C2   |   C6  |
            Reg3    |  C3   |   C7  |
            inc: 1 * majorStep + minor = 5 Col
            */
        }

        // Cumulative iteration offset
        __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration)
        {
            // TODO: Use constexpr if when C++ 17
            if(Traits::LargeDim)
            {
                constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                constexpr int32_t BlockDimSegsModMask = LsbMask<Traits::Log2BlockDimSegs>::value;

                // Cumulative offsetX = (iteration / VWSegs) % BlockDimSegs * MaxElementsPerIO + (iteration % VWSegs) * VW,
                // Cumulative offsetY = iteration / TotalSegs;
                return std::make_pair(
                    (iteration << (Traits::Log2MaxElementsPerIO - Traits::Log2VWSegs)) & (BlockDimSegsModMask << Traits::Log2MaxElementsPerIO) + 
                    (iteration & VWSegsModMask) << Traits::Log2VW,
                    iteration >> Traits::Log2TotalSegs);
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

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK); // Next block
        }
    };

    /**
 * \ingroup dataLayouts
 * @{
 */
    /**
 * *Matrix layout:*
 *
 * BlockDim = row size
 *
 * BlockK = row count
 *
 * VW = actual vector width
 *
 * N = max vector width
 *
 *
 *      (0, 0)                 (0, BlockDim - 1)
 *      v______________  ...  _v__
 *      |__________R0__  ...  ____|
 *      |__________R1__  ...  ____|
 *      |__________R2__  ...  ____|
 *      |          ...   ...      |
 *      |__________Rk__  ...  ____|
 *      ^(BlockK - 1, 0)       ^(BlockK - 1, BlockDim - 1)
 *
 * *Register layout:*
 *
 * *Guarantees:*
 *
 * RowNT guarantees the following register format, regardless of VW and data layout.
 *
 * Register 0 to contain Rows (i % N) == 0
 *
 * Register 1 to contain Rows (i % N) == 1
 *
 * Register 2 to contain Rows (i % N) == 2
 *
 * ...
 *
 * *Limitations:*
 *
 * row_major data format is not supported for VW > 1, as it produces
 * incorrect mapping
 *
 * KPerIO == # of rows per IO (either load or store)
 *
 * E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = col_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  R0   |   R4  |
 *       ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 4, N = 4, KPerIO = 8, DataT = f32, DataLayout = col_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *      Reg0    |  R0   |   R4  |
 *      Reg1    |  R1   |   R5  |
 *      Reg2    |  R2   |   R6  |
 *      Reg3    |  R3   |   R7  |
 *      ...       ...      ...
 *
 *
 * E.g. BlockDim = 32, VW = 1, N = 4, KPerIO = 2, DataT = f32, DataLayout = row_major
 *
 *      Elements 0.....31 32.....64
 *               _______________
 *       Reg0    |  R0   |   R4  |
 *        ...       ...      ...
 *
 * @note
 * E.g. BlockDim = 32, VW = 4,  N = 4, KPerIO = 8, DataT = f32, DataLayout = row_major
 * Is NOT implemented due to incorrect mapping with row_major and VW = 4
 *
*/
    ////////////// RowNT /////////////////////////
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    struct RowNT
    {
        // RowNT is orthogonal to ColNT, therefore we can use reversed coordinates
        // and opposite DataLayout from ColNT
        struct Traits
        {
            using OrthoLayout = ColNT<BlockDim,
                                      BlockK,
                                      DataT,
                                      std::conditional_t<std::is_same<DataLayout, row_major>::value,
                                                         col_major,
                                                         row_major>,
                                      VectorWidth,
                                      MaxVectorWidth>;

            enum : uint32_t
            {
                // This is the minimum K needed to correctly implement this layout.
                // Based on MaxVectorWidth due to iteration model.
                MinK       = OrthoLayout::Traits::MinK,
                MinIOCount = OrthoLayout::Traits::MinIOCount
            };

            using MappingUtil  = MappingUtil<BlockK, BlockDim, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        static_assert(!(std::is_same<DataLayout, row_major>::value && VectorWidth > 1),
                      "RowNT in row major does not support VectorWidth > 1");

        static_assert(BlockDim < AMDGCN_WAVE_SIZE,
                      "RowNT only supports BlockDim <= AMDGCN_WAVE_SIZE");

        static_assert(BlockK >= MaxVectorWidth, "BlockK must be at least MaxVectorWidth");

        static_assert(BlockK % MaxVectorWidth == 0, "BlockK must be a multiple of MaxVectorWidth");

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            // Orthogonalize coord
            return std::swap(Traits::OrthoLayout::baseOffset());
        }

        __device__ static inline typename Traits::MatrixCoordT offsetIncrement(uint32_t iteration)
        {
            // Orthogonalize coord
            return std::swap(Traits::OrthoLayout::offsetIncrement(iteration));
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            // Orthogonalize coord
            return std::swap(Traits::OrthoLayout::blockOffset());
        }

        __device__ static inline uint32_t baseDataOffset(uint32_t ldm)
        {
            return Traits::MappingUtil::dataOffset(ldm, baseOffset());
        }

        __device__ static inline uint32_t dataOffsetIncrement(uint32_t iteration, uint32_t ldm)
        {
            return Traits::MappingUtil::dataOffset(ldm, offsetIncrement(iteration));
        }

        __device__ static inline uint32_t dataBlockOffset(uint32_t ldm)
        {
            return Traits::MappingUtil::dataOffset(ldm, blockOffset());
        }
    };

} // namespace Layout

#endif // WMMA_LAYOUT_H
