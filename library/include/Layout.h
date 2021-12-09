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
 *               _______________
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
 *               _______________
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
                // Minimum K to implement this layout
                // MinK = ceilDiv(AMDGCN_WAVE_SIZE, BlockDim) * VectorWidth,
                MinK = ceilDiv(AMDGCN_WAVE_SIZE
                                   * std::max(VectorWidth, (uint32_t)PackTraits<DataT>::PackRatio),
                               BlockDim),

                // Minimum IOCount to implement this layout.
                // Iteration offsets may have periods > 1
                MinIOCount = ceilDiv(BlockDim, (uint32_t)IOTraits::ThreadsPerIO)
            };

            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, row_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            return std::make_pair(
                threadIdx.x % std::min((uint32_t)BlockDim, (uint32_t)IOTraits::ThreadsPerIO),
                ((threadIdx.x / BlockDim) * VectorWidth)
                    % std::max((uint32_t)IOTraits::KPerIO, (uint32_t)1));
        }

        __device__ static inline typename Traits::MatrixCoordT offsetIncrement(uint32_t iteration)
        {
            if(BlockDim >= IOTraits::ThreadsPerIO)
            {
                return ((iteration + 1) % Traits::MinIOCount)
                           ? std::make_pair(IOTraits::ThreadsPerIO, 0)
                           : std::make_pair(-IOTraits::ThreadsPerIO * (Traits::MinIOCount - 1),
                                            (int32_t)IOTraits::KPerIO);
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

        // __device__ static inline uint32_t dataBlockOffset(uint32_t blocks, uint32_t ldm)
        // {
        //     auto boffset = blockOffset();
        //     std::get<1>(boffset) *= blocks;
        //     return Traits::MappingUtil::dataOffset(ldm, boffset);
        // }
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth>
    struct Col<BlockDim, BlockK, DataT, col_major, VectorWidth>
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // Minimum K to implement this layout.
                //MinK = ceilDiv(AMDGCN_WAVE_SIZE * VectorWidth, BlockDim),
                MinK = ceilDiv(AMDGCN_WAVE_SIZE
                                   * std::max(VectorWidth, (uint32_t)PackTraits<DataT>::PackRatio),
                               BlockDim),

                // Minimum IOCount to implement this layout.
                // Iteration offsets may have periods > 1
                MinIOCount = ceilDiv(BlockDim, (uint32_t)IOTraits::ElementsPerIO)
            };

            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, col_major>;
            using MatrixCoordT = typename MappingUtil::CoordT;

            static_assert(MinK >= 1, "MinK must be >= 1");
            static_assert(BlockK >= MinK, "MinK not satisfied");
            static_assert(BlockK % MinK == 0, "BlockK must be divisible by MinK");
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            return std::make_pair(
                (threadIdx.x * VectorWidth)
                    % std::min((uint32_t)BlockDim, (uint32_t)IOTraits::ElementsPerIO),
                ((threadIdx.x * VectorWidth) / BlockDim)
                    % std::max((uint32_t)IOTraits::KPerIO, (uint32_t)1));
        }

        __device__ static inline typename Traits::MatrixCoordT offsetIncrement(uint32_t iteration)
        {
            // TODO: constexpr if c++17
            // Silence DIV/0 warning, remove the need for std::max
            if(BlockDim > IOTraits::ElementsPerIO)
            {
                return ((iteration + 1) % Traits::MinIOCount)
                           ? std::make_pair(IOTraits::ElementsPerIO, 0)
                           : std::make_pair(-IOTraits::ElementsPerIO * (Traits::MinIOCount - 1), 1);
            }
            else
            {
                return std::make_pair((uint32_t)0, IOTraits::KPerIO); // Shift K
            }
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK); // Shift K
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

        // __device__ static inline uint32_t dataBlockOffset(uint32_t blocks, uint32_t ldm)
        // {
        //     auto boffset = blockOffset();
        //     std::get<1>(boffset) *= blocks;
        //     return Traits::MappingUtil::dataOffset(ldm, boffset);
        // }
    };

    /**
 * \ingroup dataLayouts
 * @{
 */
    /**
 *
 * *Matrix layout:*
 *
 * Common usage: Matrix B, C
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
                MinK       = OrthoLayout::Traits::MinK,
                MinIOCount = OrthoLayout::Traits::MinIOCount
            };

            using MappingUtil  = MappingUtil<BlockK, BlockDim, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            return std::swap(Traits::OrthoLayout::baseOffset());
        }

        __device__ static inline typename Traits::MatrixCoordT offsetIncrement(uint32_t iteration)
        {
            return std::swap(Traits::OrthoLayout::offsetIncrement(iteration));
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
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

        // __device__ static inline uint32_t dataBlockOffset(uint32_t blocks, uint32_t ldm)
        // {
        //     auto boffset = blockOffset();
        //     std::get<0>(boffset) *= blocks;
        //     return Traits::MappingUtil::dataOffset(ldm, boffset);
        // }
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
    struct ColNT
    {
        using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                // This is the minimum K needed to correctly implement this layout.
                // Based on MaxVectorWidth due to iteration model.
                MinK = ceilDiv(IOTraits::ThreadsPerIO * MaxVectorWidth, BlockDim),

                BlockDimSegs = ceilDiv(BlockDim, (uint32_t)IOTraits::ThreadsPerIO),

                MinIOCount = (MaxVectorWidth / VectorWidth) * BlockDimSegs
            };
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        static_assert(!(std::is_same<DataLayout, col_major>::value && VectorWidth > 1),
                      "ColNT in column major does not support VectorWidth > 1");

        static_assert(BlockDim <= AMDGCN_WAVE_SIZE,
                      "ColNT only supports BlockDim <= AMDGCN_WAVE_SIZE");

        __device__ static inline typename Traits::MatrixCoordT baseOffset()
        {
            return std::make_pair(threadIdx.x % BlockDim,
                                  ((threadIdx.x / BlockDim) * MaxVectorWidth) % Traits::MinK);
        }

        __device__ static inline typename Traits::MatrixCoordT offsetIncrement(uint32_t iteration)
        {
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

            static_assert(IOTraits::KPerIO <= Traits::MinK, "");
            constexpr auto fullLoad = (IOTraits::KPerIO == Traits::MinK);

            constexpr auto majorStepSize = fullLoad ? Traits::MinK : Traits::MinK - MaxVectorWidth;
            constexpr auto minorStepSize = fullLoad ? 0 : VectorWidth;
            auto           doMajorStep   = ((iteration + 1) % (MaxVectorWidth / VectorWidth)) == 0;

            return std::make_pair(
                0, static_cast<uint32_t>(doMajorStep) * majorStepSize + minorStepSize);
        }

        __device__ static inline typename Traits::MatrixCoordT blockOffset()
        {
            return std::make_pair(0, BlockK); // Next block
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

        // __device__ static inline uint32_t dataBlockOffset(uint32_t blocks, uint32_t ldm)
        // {
        //     auto boffset = blockOffset();
        //     std::get<1>(boffset) *= blocks;
        //     return Traits::MappingUtil::dataOffset(ldm, boffset);
        // }
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

        // __device__ static inline uint32_t dataBlockOffset(uint32_t blocks, uint32_t ldm)
        // {
        //     auto boffset = blockOffset();
        //     std::get<0>(boffset) *= blocks;
        //     return Traits::MappingUtil::dataOffset(ldm, boffset);
        // }
    };

} // namespace Layout

#endif // WMMA_LAYOUT_H
