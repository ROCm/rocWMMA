/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_IO_SHAPE_HPP
#define ROCWMMA_IO_SHAPE_HPP

#include "config.hpp"
#include "constants.hpp"
#include "io_traits.hpp"
#include "layout.hpp"
#include "types.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename MatrixT,
                  uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t WaveCount = 1u,
                  uint32_t TestWidth
                  = 4u * Constants::AMDGCN_DWORD_SIZE_BYTES / (uint32_t)sizeof(DataT)>
        struct MaxVWSelector
        {

        private:
            enum : uint32_t
            {
                // For small block sizes (16, 32):
                // Best to keep MaxVW high and reduce splits amongst waves.
                WaveCountFactor = (BlockDim <= 32) ? 1u : WaveCount,

                // Total number of elements in a single I/O operation
                ElementsPerIO = Constants::AMDGCN_WAVE_SIZE * TestWidth * WaveCountFactor,

                // Total number of elements for the entire block
                ElementCount = BlockDim * BlockK,

                // Ensure that for MaxVW:
                // - A minimum of one IO from each wave can fit
                // - A balanced multiple of IOs from each wave
                ElementCountTest
                = (ElementsPerIO <= ElementCount) && (ElementCount % ElementsPerIO == 0),

                // TODO: When Col / Row layouts come into effect:
                // MaxVW contiguous element MUST at least fit inside block dims:
                // Matrix A:
                // - Col major: MaxVW <= BlockDim
                // - Row major: MaxVW <= BlockK
                // Matrix B / Accumulator:
                // - Col Major: MaxVW <= BlockK
                // - Row Major: MaxVW <= BlockDim
                // LeadingDimTest
                // = (std::is_same_v<MatrixT, matrix_a>
                //        && (std::is_same_v<DataLayoutT, col_major> && (TestWidth <= BlockDim))
                //    || (std::is_same_v<DataLayoutT, row_major> && (TestWidth <= BlockK)))
                //   || ((std::is_same_v<MatrixT, matrix_b> || std::is_same_v<MatrixT, accumulator>)&&(
                //           std::is_same_v<DataLayoutT, row_major> && (TestWidth <= BlockDim))
                //       || (std::is_same_v<DataLayoutT, col_major> && (TestWidth <= BlockK))),

                // Currently, all layouts are using ColOrthoVW. This means that VW must be less than BlockK
                LeadingDimTest = (TestWidth <= BlockK),

                MaxVectorWidth = (bool)ElementCountTest && (bool)LeadingDimTest
                                     ? TestWidth
                                     : MaxVWSelector<MatrixT,
                                                     BlockDim,
                                                     BlockK,
                                                     DataT,
                                                     DataLayoutT,
                                                     WaveCount,
                                                     TestWidth / 2>::Result,
            };

        public:
            enum : uint32_t
            {
                Result = (uint32_t)MaxVectorWidth
            };
        };

        template <typename MatrixT,
                  uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t WaveCount>
        struct MaxVWSelector<MatrixT, BlockDim, BlockK, DataT, DataLayoutT, WaveCount, 0u>
        {
            enum : uint32_t
            {
                Result = 1u
            };
        };

    } // namespace detail

    /*! \struct IOShape
 *  \brief Definition of rocWMMA data and matrix mapping utilities
 *         in specific matrix context.
 *
 * @tparam MatrixT fragment context
 * @tparam BlockM/N/K block dimensions
 * @tparam DataT data type
 * @tparam DataLayoutT in-memory layout as col_major or row_major
 */
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape;

    template <typename MatrixT,
              uint32_t BlockDim,
              uint32_t KDim,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct IOLayout;

    template <uint32_t BlockDim,
              uint32_t KDim,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct IOLayout<matrix_a, BlockDim, KDim, DataT, DataLayoutT, WaveCount>
    {
        // Vector size properties
        enum : uint32_t
        {
            MaxVW = detail::MaxVWSelector<matrix_a, BlockDim, KDim, DataT, DataLayoutT, WaveCount>::
                Result,
            VW = std::is_same<DataLayoutT, row_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template ColNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(std::is_same_v<DataLayoutT, col_major> && VW > 1),
                      "matrix_a in col_major currently does not support VW > 1");
    };

    template <uint32_t BlockDim,
              uint32_t KDim,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct IOLayout<matrix_b, BlockDim, KDim, DataT, DataLayoutT, WaveCount>
    {
        // Vector size properties
        enum : uint32_t
        {
            MaxVW = detail::MaxVWSelector<matrix_b, BlockDim, KDim, DataT, DataLayoutT, WaveCount>::
                Result,
            VW = std::is_same<DataLayoutT, col_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template RowNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(std::is_same_v<DataLayoutT, row_major> && VW > 1),
                      "matrix_b in row_major currently does not support VW > 1");
    };

    template <uint32_t BlockDim,
              uint32_t KDim,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct IOLayout<accumulator, BlockDim, KDim, DataT, DataLayoutT, WaveCount>
    {
        // Vector size properties
        enum : uint32_t
        {
            MaxVW = (std::is_same<DataT, float64_t>::value || ROCWMMA_ARCH_GFX11) ? 1u : 4u,
            VW    = std::is_same<DataLayoutT, col_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template RowNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(std::is_same<DataLayoutT, row_major>::value && VW > 1),
                      "accumulator in row_major currently does not support VW > 1");
    };

    template <uint32_t BlockDim, uint32_t KDim, typename DataT, uint32_t WaveCount>
    struct IOLayout<accumulator, BlockDim, KDim, DataT, void, WaveCount>
    {
        // No layout mapping without VW, MaxVW and DataLayoutT info
    };

    /************************************************
 * Matrix A default configuration: ColNT
 *
 * Dimensions: (rows x cols) = BlockDim x BlockK
 *
 * Matrix Layout:
 * BlockDim = column size
 * BlockK = column count
 *
 *  kDim ->
 *   (0, 0)              (0, BlockK - 1)
 *   v______________  ... v____
 *   |    |    |          |    |
 *   |    |    |          |    |
 *   | C0 | C1 | C2       | Ck |
 *   |    |    |          |    |
 *   |___ |___ |____  ... |____|
 *   ^(BlockDim - 1, 0)   ^(BlockDim - 1, BlockK - 1)
 *
 *  Register layout:
 *  N = Max VectorWidth
 *  Elements 0...........................64
 *            __________________
 *  Reg0     |  C0    |  CN+0   |  ...
 *  Reg1     |  C1    |  CN+1   |  ...
 *  Reg2     |  C2    |  CN+2   |  ...
 *   ...        ...       ...      ...
 *  RegN-1   |  CN-1  |  C2N-1  |  ...
 *   ...        ...       ...      ...
 *
 * For as many groups of N registers to hold BlockDim x BlockK elements.
 *
 ***********************************************/
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<matrix_a, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockK,

            BlockDim = BlockM,
            KDim     = BlockK,
        };
    };

    /************************************************
 * Matrix B default configuration: RowNT
 *
 * Dimensions: (rows x cols) = BlockK x BlockDim
 *
 * Matrix Layout:
 * BlockDim = row size
 * BlockK = row count
 *
 *  kDim    (0, 0)                 (0, BlockDim - 1)
 *    |     v______________  ...  _v__
 *    v     |__________R0__  ...  ____|
 *          |__________R1__  ...  ____|
 *          |__________R2__  ...  ____|
 *          |          ...   ...      |
 *          |__________Rk__  ...  ____|
 *          ^(BlockK - 1, 0)       ^(BlockK - 1, BlockDim - 1)
 *
 *
 *  Register layout:
 *  N = Max VectorWidth
 *  Elements 0...........................64
 *            __________________
 *  Reg0     |  R0    |  RN+0   |  ...
 *  Reg1     |  R1    |  RN+1   |  ...
 *  Reg2     |  R2    |  RN+2   |  ...
 *   ...        ...       ...      ...
 *  RegN-1   |  RN-1  |  R2N-1  |  ...
 *   ...        ...       ...      ...
 *
 * For as many groups of N registers to hold BlockDim x BlockK elements.
 *
 ***********************************************/
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<matrix_b, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockK,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockK,
        };
    };

    /************************************************
 * Matrix C/D (accumulator) default configuration:
 * RowNT, MaxVW = 4
 *
 * Dimensions: (rows x cols) = BlockK x BlockDim
 *
 * Matrix Layout:
 * BlockDim = row size
 * BlockK = row count
 *
 *  kDim    (0, 0)                 (0, BlockDim - 1)
 *    |     v______________  ...  _v__
 *    v     |__________R0__  ...  ____|
 *          |__________R1__  ...  ____|
 *          |__________R2__  ...  ____|
 *          |          ...   ...      |
 *          |__________Rk__  ...  ____|
 *          ^(BlockK - 1, 0)       ^(BlockK - 1, BlockDim - 1)
 *
 *
 *  Register layout:
 *  4 = Fixed MaxVectorWidth
 *  Elements 0...........................64
 *            ________________
 *  Reg0     |  R0    |  R4   |  ...
 *  Reg1     |  R1    |  R5   |  ...
 *  Reg2     |  R2    |  R6   |  ...
 *  Reg3     |  R3    |  R7   |  ...
 *   ...        ...       ...      ...
 *
 * For as many groups of 4 registers to hold BlockDim x BlockK elements.
 *
 ***********************************************/
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<accumulator, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockM,
        };
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_SHAPE_HPP
