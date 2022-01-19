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
#ifndef WMMA_LAYOUT_H
#define WMMA_LAYOUT_H

#include "Types.h"
#include <tuple>

// FWD decl.
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth>
struct amdgcn_io_traits;

namespace rocwmma
{
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    struct MappingUtil;
} // namespace rocwmma

#include "Layout_impl.h"

/**
 * \ingroup wmma
 * \defgroup matrixLayouts
 *
 * @brief Definition and metadata on supported matrix layouts.
 *
 * These layouts are based in matrix coordinate space. They map each of the wavefront lanes
 * into corresponding (X , Y) = (row,  col) coordinates for a particular memory layout.
 *
 * Layouts are based on an iterative indexing model.
 *
 * Three matrix offsets are calculated:
 * - Base offset: initial thread offset for layout pattern
 *
 * - Incremental offset: the thread offset increment for next iteration
 *
 * - Cumulative offset: the cumulative offset for a particular iteration,
 *   E.g. Sum of incremental offsets for [i = 0, iteration)
 */

/**
 * Layout
 */
namespace Layout
{
    /**
     * \ingroup matrixLayouts
     * @{
     */
    /**
     * ColNT Layout
     *
     * The ColNT layout will align contiguous threads to matrix columns,
     * which map to contiguous in-register column data. The 'NT' signifies
     * this mapping is identical for both col_major (N) or row_major (T)
     * data layouts. The in-register data locality is favorable for MFMA.
     *
     * - Column Height = BlockDim
     * - Column Count = BlockK
     *
     * Matrix Coords:
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
     * Column order mapping to registers is affected by MaxVectorWidth
     * and BlockDim.
     *
     * Register Mapping (BlockDim < 64):
     *
     *      N = Max VectorWidth
     *
     *               (BlockDim)
     *      Elements |0.......|.........|.....63|
     *                __________________________
     *      Reg0     |  C0    |  CN+0   |  ...  |
     *      Reg1     |  C1    |  CN+1   |  ...  |
     *      Reg2     |  C2    |  CN+2   |  ...  |
     *       ...     |  ...   |   ...   |  ...  |
     *      RegN-1   |  CN-1  |  C2N-1  |  ...  |
     *       ...        ...       ...      ...
     *
     * Register Mapping (BlockDim == 64):
     *
     *      N = Max Vector Width
     *
     *               (BlockDim)
     *      Elements |0.....63|
     *                ________
     *      Reg0     |  C0    |
     *      Reg1     |  C1    |
     *      Reg2     |  C2    |
     *       ...     |  ...   |
     *      RegN     |  CN    |
     *       ...        ...
     *
     * Register Mapping (BlockDim > 64):
     *
     *      Priority 1: Visit MaxVW Segments
     *      Priority 2: Visit BlockDim Segments
     *
     *      N = Max Vector Width
     *      S = Segment count = BlockDim / 64
     *      CX_Y: Column X, BlockDim Segment Y
     *
     *      Elements |0......63|
     *                ________
     *      Reg0     |  C0_0   |
     *      Reg1     |  C1_0   |
     *      Reg2     |  C2_0   |
     *       ...     |   ...   |
     *      RegN-1   |  CN-1_0 |
     *      RegN+0   |  C0_1   |
     *      RegN+1   |  C1_1   |
     *       ...     |   ...   |
     *      RegN*S   |  CN_S   |
     *       ...     |   ...   |
     */

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    struct ColNT : public std::conditional_t<
                       std::is_same<DataLayout, col_major>::value,
                       detail::ColOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>,
                       detail::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
    {
        struct Traits
        {
            using MappingUtil  = rocwmma::MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // ColNT enforces consistent in-register alignment of contiguous matrix column
            // elements in both row_major or col_major data layouts.
            // This layout cannot support for VW > 1 in col_major data layout otherwise the
            // ordering is broken.
            static_assert(!(std::is_same<DataLayout, col_major>::value && VectorWidth > 1),
                          "ColNT in col_major does not support VectorWidth > 1");
        };
    };

    /**
     * \ingroup matrixLayouts
     * @{
     */
    /**
     * RowNT Layout
     *
     * The RowNT layout will align contiguous threads to matrix rows,
     * which map to contiguous in-register row data. The 'NT' signifies
     * this mapping is identical for both col_major (N) or row_major (T)
     * data layouts. The in-register data locality is favorable for MFMA.
     *
     * - Row Width = BlockDim
     * - Row Count = BlockK
     *
     * Matrix Coords:
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
     * Row order mapping to registers is affected by MaxVectorWidth
     * and BlockDim.
     *
     * Register Mapping (BlockDim < 64):
     *
     *      N = Max VectorWidth
     *
     *               (BlockDim)
     *      Elements |0.......|.........|.....63|
     *                __________________________
     *      Reg0     |  R0    |  RN+0   |  ...  |
     *      Reg1     |  R1    |  RN+1   |  ...  |
     *      Reg2     |  R2    |  RN+2   |  ...  |
     *       ...     |  ...   |   ...   |  ...  |
     *      RegN-1   |  RN-1  |  R2N-1  |  ...  |
     *       ...        ...       ...      ...
     *
     * Register Mapping (BlockDim == 64):
     *
     *      N = Max Vector Width
     *
     *               (BlockDim)
     *      Elements |0.....63|
     *                ________
     *      Reg0     |  R0    |
     *      Reg1     |  R1    |
     *      Reg2     |  R2    |
     *       ...     |  ...   |
     *      RegN     |  RN    |
     *       ...        ...
     *
     * Register Mapping (BlockDim > 64):
     *
     *      Priority 1: Visit MaxVW Segments
     *      Priority 2: Visit BlockDim Segments
     *
     *      N = Max Vector Width
     *      S = Segment count = BlockDim / 64
     *      RX_Y: Row X, BlockDim Segment Y
     *
     *      Elements |0......63|
     *                ________
     *      Reg0     |  R0_0   |
     *      Reg1     |  R1_0   |
     *      Reg2     |  R2_0   |
     *       ...     |   ...   |
     *      RegN-1   |  RN-1_0 |
     *      RegN+0   |  R0_1   |
     *      RegN+1   |  R1_1   |
     *       ...     |   ...   |
     *      RegN*S   |  RN_S   |
     *       ...     |   ...   |
     */
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    struct RowNT : public std::conditional_t<
                       std::is_same<DataLayout, col_major>::value,
                       detail::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                       detail::RowOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>>
    {
        struct Traits
        {
            using MappingUtil  = rocwmma::MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // RowNT enforces consistent in-register alignment of contiguous matrix row
            // elements in both in row_major or col_major data layouts.
            // This layout cannot support for VW > 1 in row_major data layout otherwise the
            // ordering is broken.
            static_assert(!(std::is_same<DataLayout, row_major>::value && VectorWidth > 1),
                          "RowNT in row_major does not support VectorWidth > 1");
        };
    };

    /**
     * \ingroup dataLayouts
     * @{
     */
    /**
     * Col Layout
     *
     * Col signifies that this layout will align contiguous threads
     * in preference of matrix columns. This is a specific case of ColNT
     * where VectorWidth = MaxVectorWidth.
     *
     * - Column Height = BlockDim
     * - Column Count = BlockK
     *
     *      Matrix Coords
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
     * In this particular layout, columns are iterated over in the kDim direction
     * until the entire block has been visited.
     *
     * The order that columns map to registers is affected by DataLayout, VectorWidth
     * and BlockDim.
     */

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth = VectorWidth>
    struct Col : public std::conditional_t<
                     std::is_same<DataLayout, col_major>::value,
                     detail::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                     detail::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
    {
        struct Traits
        {
            using MappingUtil  = rocwmma::MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;
        };
    };

    /**
     * \ingroup dataLayouts
     * @{
     */
    /**
     * Row Layout
     *
     * Row signifies that this layout will align contiguous threads
     * in preference of matrix rows. This is a specific case of RowNT
     * where VectorWidth = MaxVectorWidth.
     *
     * - Row Width = BlockDim
     * - Row Count = BlockK
     *
     *      Matrix Coords
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
     * In this particular layout, rows are iterated over in the kDim direction
     * until the entire block has been visited.
     *
     * The order that rows map to registers is affected by DataLayout, VectorWidth
     * and BlockDim.
     */
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth = VectorWidth>
    struct Row : public std::conditional_t<
                     std::is_same<DataLayout, row_major>::value,
                     detail::RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                     detail::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
    {
        struct Traits
        {
            using MappingUtil  = rocwmma::MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;
        };
    };

} // namespace Layout

#endif // WMMA_LAYOUT_H
