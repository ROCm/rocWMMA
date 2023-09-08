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
#ifndef ROCWMMA_LAYOUT_HPP
#define ROCWMMA_LAYOUT_HPP

#include "layout_impl.hpp"

namespace rocwmma
{
    namespace DataLayout
    {
        template <typename DataOrientation>
        using Array1d = typename ::rocwmma::detail::template DataSpace<DataOrientation>;

        using RowMajor = Array1d<row_major>;
        using ColMajor = Array1d<col_major>;

    } // namespace DataLayout

    namespace MatrixLayout
    {
        /**
         * \defgroup Matrix_Layouts Matrix Layouts
         *
         * @brief Definition and metadata on supported matrix layouts.
         * @{
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
         * ColNT and RowNT layouts: The MFMA alignments. Matrix core instructions
         * have specific mapping for inputs and outputs: inputs being contiguous
         * row / col data, and outputs being contiguous row / col data depending
         * on the order of inputs supplied. ColNT and RowNT layouts are orthogonal
         * to each other in the opposite data layout.
         *
         * Out of the box, the rocWMMA API needs to support MFMA input / output
         * layouts without needing to re-order data in LDS first. At a cost to
         * some performance, we can guarantee the rocWMMA loads and stores  of
         * fragments are mapped identically, whether the data is in row_major or
         * col_major, whatever the vector width (with restrictions) and regardless
         * of their datatype. ColNT and RowNT represent consistent matrix-to-register
         * mapping supporting the wide variety of configurations supported by the
         * rocWMMA API.
         *
         * Performance considerations:
         * Inline vector width configurations (e.g. col_major in ColNT and row_major
         * in RowNT) are restricted to VW=1 such that matrix-to-register mapping
         * is preserved. This layout is better suited to smaller block sizes that
         * require MFMA alignment, such as the rocwmma::mma_sync multiplication
         * and accumulation.
         *
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
         * First priority: columns are mapped to registers as
         * if VW = MaxVW. If VW < MaxVW, mapping will iterate up to MaxVW in
         * segments of VW prior to moving onto the next block. This ensures
         * register mapping is preserved for VW up to MaxVW. As a result,
         * this also helps facilitate data transposes as orthogonal mappings
         * may adjust VW without changing the overall order.
         *
         * Second priority: BlockDim.
         *
         * If BlockDim < 64, then multiple full columns
         * will be mapped to each register. The column ordering however will
         * be determined as a multiple of MaxVW to fulfill MaxVW priority.
         *
         *    Register Mapping (BlockDim < 64):
         *      N = Max VectorWidth
         *
         *               | BlockDim |
         *      Elements |0.........|.........|.....63|
         *                __________________________
         *      Reg0     |  C0      |  CN+0   |  ...  |
         *      Reg1     |  C1      |  CN+1   |  ...  |
         *      Reg2     |  C2      |  CN+2   |  ...  |
         *       ...     |  ...     |   ...   |  ...  |
         *      RegN-1   |  CN-1    |  C2N-1  |  ...  |
         *       ...        ...         ...      ...
         *
         * If BlockDim = 64, then a single full column will be mapped
         * contiguously to each register. Column ordering still fufills MaxVW
         * priority first.
         *
         *    Register Mapping (BlockDim == 64):
         *
         *      N = Max Vector Width
         *
         *               |BlockDim|
         *      Elements |0.....63|
         *                ________
         *      Reg0     |  C0    |
         *      Reg1     |  C1    |
         *      Reg2     |  C2    |
         *       ...     |  ...   |
         *      RegN-1   |  CN-1  |
         *       ...        ...
         *
         * If BlockDim > 64, BlockDim segments are mapped contiguously to
         * registers. Mapping ordering fills registers with VW segments first
         * to satisfy MaxVW priority before moving on to the next BlockDim
         * segment.
         *
         *    Register Mapping (BlockDim > 64):
         *
         *      Priority 1: Visit MaxVW Segments
         *      Priority 2: Visit BlockDim Segments
         *
         *      N = Max Vector Width
         *      S = Segment count = BlockDim / 64
         *      (X, Y): Column X, BlockDim Segment Y
         *
         *      Elements |0......63|
         *                __________
         *      Reg0     |  (0, 0)  |
         *      Reg1     |  (1, 0)  |
         *      Reg2     |  (2, 0)  |
         *       ...     |   ...    |
         *      RegN-1   | (N-1, 0) |
         *      RegN+0   | (0, 1)   |
         *      RegN+1   | (1, 1)   |
         *       ...     |   ...    |
         *      RegN*S   |  (N, S)  |
         *       ...     |   ...    |
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
                using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
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
         * First priority: rows are mapped to registers as
         * if VW = MaxVW. If VW < MaxVW, mapping will iterate up to MaxVW in
         * segments of VW prior to moving onto the next block. This ensures
         * register mapping is preserved for VW up to MaxVW. As a result,
         * this also helps facilitate data transposes as orthogonal mappings
         * may adjust VW without changing the overall order.
         *
         * Second priority: BlockDim.
         *
         * If BlockDim < 64, then multiple full rows
         * will be mapped to each register. The row ordering however will
         * be determined as a multiple of MaxVW to fulfill MaxVW priority.
         *
         *  Register Mapping (BlockDim < 64):
         *
         *      N = Max VectorWidth
         *
         *               |BlockDim|
         *      Elements |0.......|.........|.....63|
         *                __________________________
         *      Reg0     |  R0    |  RN+0   |  ...  |
         *      Reg1     |  R1    |  RN+1   |  ...  |
         *      Reg2     |  R2    |  RN+2   |  ...  |
         *       ...     |  ...   |   ...   |  ...  |
         *      RegN-1   |  RN-1  |  R2N-1  |  ...  |
         *       ...        ...       ...      ...
         *
         * If BlockDim = 64, then a single full row will be mapped
         * contiguously to each register. row ordering still fufills MaxVW
         * priority first.
         *
         *   Register Mapping (BlockDim == 64):
         *
         *      N = Max Vector Width
         *
         *               |BlockDim|
         *      Elements |0.....63|
         *                ________
         *      Reg0     |  R0    |
         *      Reg1     |  R1    |
         *      Reg2     |  R2    |
         *       ...     |  ...   |
         *      RegN-1   |  RN-1  |
         *       ...        ...
         *
         * If BlockDim > 64, BlockDim segments are mapped contiguously to
         * registers. Mapping ordering fills registers with VW segments first
         * to satisfy MaxVW priority before moving on to the next BlockDim
         * segment.
         *
         *  Register Mapping (BlockDim > 64):
         *
         *      Priority 1: Visit MaxVW Segments
         *      Priority 2: Visit BlockDim Segments
         *
         *      N = Max Vector Width
         *      S = Segment count = BlockDim / 64
         *      (X, Y): Row X, BlockDim Segment Y
         *
         *      Elements |0......63|
         *                __________
         *      Reg0     |  (0, 0)  |
         *      Reg1     |  (1, 0)  |
         *      Reg2     |  (2, 0)  |
         *       ...     |   ...    |
         *      RegN-1   | (N-1, 0) |
         *      RegN+0   | (0, 1)   |
         *      RegN+1   | (1, 1)   |
         *       ...     |   ...    |
         *      RegN*S   |  (N, S)  |
         *       ...     |   ...    |
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
                using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
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
         * Col and Row layouts: unrestricted vector width col and row ordering.
         * Unline their NT cousins, these layouts do not guarantee the same
         * matrix-to-register mappings for both row_major and col_major data layouts
         * needed for MFMA instruction alignment. They do however implement the same
         * ordering priorities which can facilitate fragment data transpose. Col and
         * Row layouts are orthogonal to each other in the opposite data layout.
         *
         * Col Layout
         *
         * Col signifies that this layout will align contiguous threads
         * in preference of matrix columns.
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
         * Column order mapping to registers is affected by data layout,
         * max vector width and BlockDim.
         *
         * First priority: columns are mapped to registers as
         * if VW = MaxVW. If VW < MaxVW, mapping will iterate up to MaxVW in
         * segments of VW prior to moving onto the next block. This ensures
         * register mapping is preserved for VW up to MaxVW. As a result,
         * this also helps facilitate data transposes as orthogonal mappings
         * may adjust VW without changing the overall order.
         *
         * Second priority: BlockDim.
         *
         * If BlockDim < 64, then multiple columns will be mapped to multiple registers
         * to a depth of MaxVW.
         * The column ordering however will be determined as a multiple of MaxVW to
         * fulfill MaxVW priority.
         *
         *  Data Layout: row_major \<same as ColNT\>
         *  Register Mapping (BlockDim < 64):
         *
         *      N = Max VectorWidth
         *
         *               | BlockDim |
         *      Elements |0.........|.........|.....63|
         *                __________________________
         *      Reg0     |  C0      |  CN+0   |  ...  |
         *      Reg1     |  C1      |  CN+1   |  ...  |
         *      Reg2     |  C2      |  CN+2   |  ...  |
         *       ...     |  ...     |   ...   |  ...  |
         *      RegN-1   |  CN-1    |  C2N-1  |  ...  |
         *       ...        ...         ...      ...
         *
         *  Data Layout: col_major
         *  Register Mapping (BlockDim < 64):
         *
         *      N = Max VectorWidth
         *      (X, Y) = Col X, Element Y
         *
         *   Elements 0......... 1 .......BlockDim / MaxVW ..............63
         *           _________________________________________________________
         *   Reg0   | (0, 0)   | (0, N)    | ... | (1, 0)   |  ...  |    ...  |
         *   Reg1   | (0, 1)   | (0, N+1)  | ... | (1, 1)   |  ...  |    ...  |
         *   ....   |  ...     |  ...      | ... | ....     |  ...  |    ...  |
         *   RegN-1 | (0, N-1) | (0, 2N-1) | ... | (1, N-1) |  ...  |    ...  |
         *
         *
         * If BlockDim = 64, then multiple full cols will be mapped
         * across multiple registers. Col ordering still fufills MaxVW
         * priority first.
         *
         *  Data Layout: row_major \<same as ColNT\>
         *  Register Mapping (BlockDim == 64):
         *
         *      N = Max Vector Width
         *
         *               |BlockDim|
         *      Elements |0.....63|
         *                ________
         *      Reg0     |  C0    |
         *      Reg1     |  C1    |
         *      Reg2     |  C2    |
         *       ...     |  ...   |
         *      RegN-1   |  CN-1  |
         *       ...        ...
         *
         *  Data Layout: col_major
         *  Register Mapping (BlockDim == 64):
         *
         *      N = Max VectorWidth
         *      (X, Y) = Col X, Element Y
         *
         *   Elements 0......... 1 ...................................... 63
         *           ______________________________________________________________
         *   Reg0   | (0, 0)   |  (0, N)    | ... | (1, 0)   | ... | (N-1, 64-N)   |
         *   Reg1   | (0, 1)   |  (0, N+1)  | ... | (1, 1)   | ... | (N-1, 64-N+1) |
         *   ....   |   ...    |    ...     | ... |    ...   | ... |    ...        |
         *   RegN-1 | (0, N-1) |  (0, 2N-1) | ... | (1, N-1) | ... | (N-1, 64-1)   |
         *
         * If BlockDim > 64, BlockDim segments are mapped contiguously to
         * registers. Mapping ordering fills registers with VW segments first
         * to satisfy MaxVW priority before moving on to the next BlockDim
         * segment.
         *
         *  Data Layout: row_major \<same as ColNT>
         *  Register Mapping (BlockDim > 64):
         *
         *      Priority 1: Visit MaxVW Segments
         *      Priority 2: Visit BlockDim Segments
         *
         *      N = Max Vector Width
         *      S = Segment count = BlockDim / 64
         *      (X, Y): Column X, BlockDim Segment Y
         *
         *      Elements |0......63|
         *                _________
         *      Reg0     | (0, 0)  |
         *      Reg1     | (1, 0)  |
         *      Reg2     | (2, 0)  |
         *       ...     |   ...   |
         *      RegN-1   | (N-1, 0)|
         *      RegN+0   | (0, 1)  |
         *      RegN+1   | (1, 1)  |
         *       ...     |   ...   |
         *      RegN*S   | (N, S)  |
         *       ...     |   ...   |
         *
         *  Data Layout: col_major
         *  Register Mapping (BlockDim > 64):
         *
         *      N = Max VectorWidth
         *      (X, Y, Z) = Col X, Segment Y, Element Z
         *
         *   Elements 0.................. 1 ..........................................63
         *            _________________________________________________________________________
         *   Reg0    | (0, 0, 0)   | (0, 0, N)    | ... | (1, 0, 0)   | ... | (N-1, 0, 64-N)   |
         *   Reg1    | (0, 0, 1)   | (0, 0, N+1)  | ... | (1, 0, 1)   | ... | (N-1, 0, 64-N+1) |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
         *   RegN-1  | (0, 0, N-1) | (0, 0, 2N-1) | ... | (1, 0, N-1) | ... | (N-1, 0, 64-1)   |
         *                          ...  Repeat for next BlockDim seg ...
         *   RegN    | (0, 1, 0)   | (0, 1, N)    | ... | (1, 1, 0)   | ... | (N-1, 1, 63-N)   |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
         *   Reg2N-1 | (0, 1, N-1) | (0, 1, 2N-1) | ... | (1, 1, N-1) | ... | (N-1, 1, 64-1)   |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
         *
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
                using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
                using MatrixCoordT = typename MappingUtil::MatrixCoordT;
            };
        };

        /**
         * Row Layout
         *
         * Row signifies that this layout will align contiguous threads
         * in preference of matrix rows.
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
         *
         * First priority: rows are mapped to registers as
         * if VW = MaxVW. If VW < MaxVW, mapping will iterate up to MaxVW in
         * segments of VW prior to moving onto the next block. This ensures
         * register mapping is preserved for VW up to MaxVW. As a result,
         * this also helps facilitate data transposes as orthogonal mappings
         * may adjust VW without changing the overall order.
         *
         * Second priority: BlockDim.
         *
         * If BlockDim < 64, then multiple rows will be mapped to multiple registers
         * to a depth of MaxVW.
         * The row ordering however will be determined as a multiple of MaxVW to
         * fulfill MaxVW priority.
         *
         *  Data Layout: col_major \<same as RowNT\>
         *  Register Mapping (BlockDim < 64):
         *
         *      N = Max VectorWidth
         *
         *               | BlockDim |
         *      Elements |0.........|.........|.....63|
         *                __________________________
         *      Reg0     |  R0      |  RN+0   |  ...  |
         *      Reg1     |  R1      |  RN+1   |  ...  |
         *      Reg2     |  R2      |  RN+2   |  ...  |
         *       ...     |  ...     |   ...   |  ...  |
         *      RegN-1   |  RN-1    |  R2N-1  |  ...  |
         *       ...        ...         ...      ...
         *
         *  Data Layout: row_major
         *  Register Mapping (BlockDim < 64):
         *
         *      N = Max VectorWidth
         *      (X, Y) = Row X, Element Y
         *
         *   Elements 0......... 1 .......BlockDim / MaxVW ..............63
         *           _________________________________________________________
         *   Reg0   | (0, 0)   | (0, N)    | ... | (1, 0)   |  ...  |    ...  |
         *   Reg1   | (0, 1)   | (0, N+1)  | ... | (1, 1)   |  ...  |    ...  |
         *   ....   |  ...     |  ...      | ... | ....     |  ...  |    ...  |
         *   RegN-1 | (0, N-1) | (0, 2N-1) | ... | (1, N-1) |  ...  |    ...  |
         *
         *
         * If BlockDim = 64, then multiple full row will be mapped
         * across multiple registers. Row ordering still fufills MaxVW
         * priority first.
         *
         *  Data Layout: col_major \<same as RowNT\>
         *  Register Mapping (BlockDim == 64):
         *
         *      N = Max Vector Width
         *
         *               |BlockDim|
         *      Elements |0.....63|
         *                ________
         *      Reg0     |  R0    |
         *      Reg1     |  R1    |
         *      Reg2     |  R2    |
         *       ...     |  ...   |
         *      RegN-1   |  RN-1  |
         *       ...        ...
         *
         *  Data Layout: row_major
         *  Register Mapping (BlockDim == 64):
         *
         *      N = Max VectorWidth
         *      (X, Y) = Row X, Element Y
         *
         *   Elements 0......... 1 ...................................... 63
         *           ______________________________________________________________
         *   Reg0   | (0, 0)   |  (0, N)    | ... | (1, 0)   | ... | (N-1, 64-N)   |
         *   Reg1   | (0, 1)   |  (0, N+1)  | ... | (1, 1)   | ... | (N-1, 64-N+1) |
         *   ....   |   ...    |    ...     | ... |    ...   | ... |    ...        |
         *   RegN-1 | (0, N-1) |  (0, 2N-1) | ... | (1, N-1) | ... | (N-1, 64-1)   |
         *               ...  Repeat for next BlockDim segments ...
         *
         * If BlockDim > 64, BlockDim segments are mapped contiguously to
         * registers. Mapping ordering fills registers with VW segments first
         * to satisfy MaxVW priority before moving on to the next BlockDim
         * segment.
         *
         *  Data Layout: col_major \<same as RowNT\>
         *  Register Mapping (BlockDim > 64):
         *
         *      Priority 1: Visit MaxVW Segments
         *      Priority 2: Visit BlockDim Segments
         *
         *      N = Max Vector Width
         *      S = Segment count = BlockDim / 64
         *      (X, Y): Row X, BlockDim Segment Y
         *
         *      Elements |0......63|
         *                _________
         *      Reg0     | (0, 0)  |
         *      Reg1     | (1, 0)  |
         *      Reg2     | (2, 0)  |
         *       ...     |   ...   |
         *      RegN-1   | (N-1, 0)|
         *      RegN+0   | (0, 1)  |
         *      RegN+1   | (1, 1)  |
         *       ...     |   ...   |
         *      RegN*S   | (N, S)  |
         *       ...     |   ...   |
         *
         *  Data Layout: row_major
         *  Register Mapping (BlockDim > 64):
         *
         *      N = Max VectorWidth
         *      (X, Y, Z) = Row X, Segment Y, Element Z
         *
         *   Elements 0.................. 1 ..........................................63
         *            _________________________________________________________________________
         *   Reg0    | (0, 0, 0)   | (0, 0, N)    | ... | (1, 0, 0)   | ... | (N-1, 0, 64-N)   |
         *   Reg1    | (0, 0, 1)   | (0, 0, N+1)  | ... | (1, 0, 1)   | ... | (N-1, 0, 64-N+1) |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
         *   RegN-1  | (0, 0, N-1) | (0, 0, 2N-1) | ... | (1, 0, N-1) | ... | (N-1, 0, 64-1)   |
         *                          ...  Repeat for next BlockDim seg ...
         *   RegN    | (0, 1, 0)   | (0, 1, N)    | ... | (1, 1, 0)   | ... | (N-1, 1, 63-N)   |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
         *   Reg2N-1 | (0, 1, N-1) | (0, 1, 2N-1) | ... | (1, 1, N-1) | ... | (N-1, 1, 64-1)   |
         *   ....    |  ...        |    ...       | ... |   ....      | ... |     ...          |
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
                using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
                using MatrixCoordT = typename MappingUtil::MatrixCoordT;
            };
        };

        template <typename LayoutT>
        using OrthogonalLayout_t = typename detail::OrthogonalLayout<LayoutT>::Type;
        /** @}*/

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_HPP
