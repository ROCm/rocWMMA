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
#ifndef ROCWMMA_IO_SHAPE_HPP
#define ROCWMMA_IO_SHAPE_HPP

#include "config.hpp"
#include "constants.hpp"
#include "io_traits.hpp"
#include "layout.hpp"
#include "types.hpp"

namespace rocwmma
{
    /*! \struct IOShape
 *  \brief Definition of rocWMMA data and matrix mapping utilities
 *         in specific matrix context.
 *
 * @tparam MatrixT fragment context
 * @tparam BlockM/N/K block dimensions
 * @tparam DataT data type
 * @tparam DataLayoutT in-memory layout as col_major or row_major
 */
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOShape;

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
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOShape<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockK,

            BlockDim = BlockM,
            KDim     = BlockK,

            MaxVectorWidth = detail::VecWidthTraits<BlockDim, KDim, DataT>::MaxVectorWidth,
            VectorWidth    = std::is_same<DataLayoutT, row_major>::value ? MaxVectorWidth : 1
        };

        static_assert(!(std::is_same<DataLayoutT, col_major>::value && VectorWidth > 1),
                      "matrix_a in col_major currently does not support VectorWidth > 1");

        using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout = MatrixLayout::
            template ColNT<BlockDim, KDim, DataT, DataLayoutT, VectorWidth, MaxVectorWidth>;
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
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOShape<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT>
    {
        enum : uint32_t
        {
            BlockHeight = BlockK,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockK,

            MaxVectorWidth = detail::VecWidthTraits<BlockDim, KDim, DataT>::MaxVectorWidth,
            VectorWidth    = std::is_same<DataLayoutT, col_major>::value ? MaxVectorWidth : 1
        };

        static_assert(!(std::is_same<DataLayoutT, row_major>::value && VectorWidth > 1),
                      "matrix_b in row_major currently does not support VectorWidth > 1");

        using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout = MatrixLayout::
            template RowNT<BlockDim, KDim, DataT, DataLayoutT, VectorWidth, MaxVectorWidth>;
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
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOShape<accumulator, BlockM, BlockN, BlockK, DataT, DataLayoutT>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockM,

            MaxVectorWidth = (std::is_same<DataT, float64_t>::value || ROCWMMA_ARCH_GFX11) ? 1 : 4,

            VectorWidth = std::is_same<DataLayoutT, col_major>::value ? MaxVectorWidth : 1,
        };

        static_assert(!(std::is_same<DataLayoutT, row_major>::value && VectorWidth > 1),
                      "accumulator in row_major currently does not support VectorWidth > 1");

        using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout = MatrixLayout::
            template RowNT<BlockDim, KDim, DataT, DataLayoutT, VectorWidth, MaxVectorWidth>;
    };

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    struct IOShape<accumulator, BlockM, BlockN, BlockK, DataT, void>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockM
        };

        // No DataLayout or MatrixLayout without VW, MaxVW and DataOrientation info
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_SHAPE_HPP
