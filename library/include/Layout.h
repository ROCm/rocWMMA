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

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
struct MappingUtil;

/**
 * \ingroup wmma
 * \defgroup dataLayouts
 *
 * @brief Definition and metadata on supported data layout of matrices.
 *
 * These layouts are based in Matrix Space. They map each of the wavefront lanes
 * into corresponding (X , Y) = (row,  col) coordinates for a particular memory layout.
 *
 * Layouts are based on an iterative indexing model such that block sizes are flexible.
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
     * \ingroup dataLayouts
     * @{
     */
    /**
     * ColNT Layout
     *
     * ColNT signifies that this layout will align contiguous threads
     * in preference of matrix columns. The 'NT' signifies that
     * given a particular Vector Width, column ordering in register will
     * match that of the full Max Vector Width over the course of iteration.
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
     * The order that columns map to registers is affected by DataLayout, MaxVectorWidth
     * and BlockDim. However it is important to note that this order is fixed for
     * 1 <= VectorWidth <= MaxVectorWidth, given MaxVectorWidth % VectorWidth = 0.
     */

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
                // Internal Meta-data, E.g.

                // Number of threads per wave
                WaveSize,

                // Flag for large BlockDim
                LargeDim,

                //...
            };

            // Matrix coords and mapping util are used to find the final data coord
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        // Matrix coord offsets
        __device__ static inline typename Traits::MatrixCoordT baseOffset();
        __device__ static inline typename Traits::MatrixCoordT
            incrementalOffset(uint32_t iteration);
        __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration);
    };

    /**
     * \ingroup dataLayouts
     * @{
     */
    /**
     *
     * RowNT Layout
     * This layout aligns contiguous threads in preference of matrix rows.
     * The 'NT' signifies that given a particular Vector Width, column ordering in register will
     * match that of the full Max Vector Width over the course of iteration.
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
     * The order that rows map to registers is affected by DataLayout, MaxVectorWidth
     * and BlockDim. However it is important to note that this order is fixed for
     * 1 <= VectorWidth <= MaxVectorWidth, given MaxVectorWidth % VectorWidth = 0.
     *
     * This layout is orthogonal to the ColNT class, so the coordinates are
     * mirrored.
     */
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

            // enum : uint32_t
            // {
            //     // This is the minimum K needed to correctly implement this layout.
            //     // Based on MaxVectorWidth due to iteration model.
            //     MinK       = OrthoLayout::Traits::MinK,
            //     MinIOCount = OrthoLayout::Traits::MinIOCount
            // };

            // Matrix coords and mapping util are used to find the final data coord
            using MappingUtil  = MappingUtil<BlockK, BlockDim, DataT, DataLayout>;
            using MatrixCoordT = typename MappingUtil::CoordT;
        };

        // Matrix coord offsets
        __device__ static inline typename Traits::MatrixCoordT baseOffset();
        __device__ static inline typename Traits::MatrixCoordT
            incrementalOffset(uint32_t iteration);
        __device__ static inline typename Traits::MatrixCoordT cumulativeOffset(uint32_t iteration);
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
              uint32_t VectorWidth>
    struct Col : public ColNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, VectorWidth>
    {
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
              uint32_t VectorWidth>
    struct Row : public RowNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, VectorWidth>
    {
    };

} // namespace Layout

#include "Layout_impl.h"

#endif // WMMA_LAYOUT_H
