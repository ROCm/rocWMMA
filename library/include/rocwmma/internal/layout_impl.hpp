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
#ifndef ROCWMMA_LAYOUT_IMPL_HPP
#define ROCWMMA_LAYOUT_IMPL_HPP

#include "io_traits.hpp"
#include "layout.hpp"
#include "mapping_util.hpp"
#include "utils.hpp"

namespace rocwmma
{
    namespace DataLayout
    {
        namespace detail
        {
            ///
            /// Helper to obtain orthogonal data layout
            ///
            template <typename LayoutT>
            struct OrthogonalLayout;

            template <>
            struct OrthogonalLayout<row_major>
            {
                using Type = col_major;
            };

            template <>
            struct OrthogonalLayout<col_major>
            {
                using Type = row_major;
            };

            ///
            /// Helper to ensure layout types are consistent (same)
            ///
            template <typename LhsDataLayout, typename RhsDataLayout>
            struct ConsistencyCheck : public std::false_type
            {
            };

            template <typename DataLayout>
            struct ConsistencyCheck<DataLayout, DataLayout> : public std::true_type
            {
            };

            ///
            /// Helper to check if layout types are orthogonal
            ///
            template <typename LhsDataLayout, typename RhsDataLayout>
            struct OrthogonalCheck : public std::true_type
            {
            };

            template <typename DataLayout>
            struct OrthogonalCheck<DataLayout, DataLayout> : public std::false_type
            {
            };

        } // namespace detail

        template <typename DataLayoutT>
        using OrthogonalLayout_t = typename detail::OrthogonalLayout<DataLayoutT>::Type;

        // TODO: C++17 OrthogonalCheck_v
        // TODO: C++17 ConsistencyCheck_v

    } // namespace DataLayout

    namespace MatrixLayout
    {
        ///
        /// Fwd declaration of matrix layouts used in API
        ///
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColNT;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowNT;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct Col;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct Row;

        namespace detail
        {
            ///
            /// Helper to obtain orthogonal data layout
            ///
            template <typename LayoutT>
            struct OrthogonalLayout;

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalLayout<
                ColNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>>
            {
                using Type = RowNT<BlockDim,
                                   BlockK,
                                   DataT,
                                   typename DataLayout::template OrthogonalLayout_t<DataLayout>,
                                   VectorWidth,
                                   MaxVectorWidth>;
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalLayout<
                RowNT<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>>
            {
                using Type = ColNT<BlockDim,
                                   BlockK,
                                   DataT,
                                   typename DataLayout::template OrthogonalLayout_t<DataLayout>,
                                   VectorWidth,
                                   MaxVectorWidth>;
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalLayout<
                Col<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>>
            {
                using Type = Row<BlockDim,
                                 BlockK,
                                 DataT,
                                 typename DataLayout::template OrthogonalLayout_t<DataLayout>,
                                 VectorWidth,
                                 MaxVectorWidth>;
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalLayout<
                Row<BlockDim, BlockK, DataT, DataLayout, VectorWidth, MaxVectorWidth>>
            {
                using Type = Col<BlockDim,
                                 BlockK,
                                 DataT,
                                 typename DataLayout::template OrthogonalLayout_t<DataLayout>,
                                 VectorWidth,
                                 MaxVectorWidth>;
            };

            ///
            /// Check for consistency in element ordering between two layouts
            ///
            template <typename LhsMatrixLayout, typename RhsMatrixLayout>
            struct ConsistencyCheck : public std::false_type
            {
            };

            // Same type is compatible
            template <typename MatrixLayout>
            struct ConsistencyCheck<MatrixLayout, MatrixLayout> : public std::true_type
            {
            };

            // ColNT and RowNT layouts guarantee a level of consistency between col / row major
            // data layouts, given some restrictions vector width and same MaxVW
            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t MaxVectorWidth,
                      uint32_t RhsVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::ColNT<BlockDim, BlockK, DataT, col_major, 1, MaxVectorWidth>,
                MatrixLayout::
                    ColNT<BlockDim, BlockK, DataT, row_major, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t MaxVectorWidth,
                      uint32_t LhsVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::
                    ColNT<BlockDim, BlockK, DataT, row_major, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::ColNT<BlockDim, BlockK, DataT, col_major, 1, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t MaxVectorWidth,
                      uint32_t LhsVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::
                    RowNT<BlockDim, BlockK, DataT, col_major, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::RowNT<BlockDim, BlockK, DataT, row_major, 1, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t MaxVectorWidth,
                      uint32_t RhsVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::RowNT<BlockDim, BlockK, DataT, row_major, 1, MaxVectorWidth>,
                MatrixLayout::
                    RowNT<BlockDim, BlockK, DataT, col_major, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            // Col and Row layouts guarantee a level of consistency between variable vector widths in
            // matching data layouts, given the same MaxVW.
            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::
                    Col<BlockDim, BlockK, DataT, DataLayout, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::
                    Col<BlockDim, BlockK, DataT, DataLayout, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename DataLayout,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct ConsistencyCheck<
                MatrixLayout::
                    Row<BlockDim, BlockK, DataT, DataLayout, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::
                    Row<BlockDim, BlockK, DataT, DataLayout, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            ///
            /// Check for layout orthogonality
            ///

            template <typename LhsMatrixLayout, typename RhsMatrixLayout>
            struct OrthogonalCheck : public std::false_type
            {
            };

            // Same type is not orthogonal
            template <typename MatrixLayout>
            struct OrthogonalCheck<MatrixLayout, MatrixLayout> : public std::false_type
            {
            };

            template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::ColNT<BlockDim, BlockK, DataT, col_major, 1, MaxVectorWidth>,
                MatrixLayout::RowNT<BlockDim, BlockK, DataT, row_major, 1, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::
                    ColNT<BlockDim, BlockK, DataT, row_major, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::
                    RowNT<BlockDim, BlockK, DataT, col_major, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::RowNT<BlockDim, BlockK, DataT, row_major, 1, MaxVectorWidth>,
                MatrixLayout::ColNT<BlockDim, BlockK, DataT, col_major, 1, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::
                    RowNT<BlockDim, BlockK, DataT, col_major, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::
                    ColNT<BlockDim, BlockK, DataT, row_major, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename LhsDataLayout,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::
                    Col<BlockDim, BlockK, DataT, LhsDataLayout, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::Row<BlockDim,
                                  BlockK,
                                  DataT,
                                  typename DataLayout::template OrthogonalLayout_t<LhsDataLayout>,
                                  RhsVectorWidth,
                                  MaxVectorWidth>> : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      typename LhsDataLayout,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::
                    Row<BlockDim, BlockK, DataT, LhsDataLayout, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::Col<BlockDim,
                                  BlockK,
                                  DataT,
                                  typename DataLayout::template OrthogonalLayout_t<LhsDataLayout>,
                                  RhsVectorWidth,
                                  MaxVectorWidth>> : public std::true_type
            {
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t LhsVectorWidth,
                      uint32_t RhsVectorWidth,
                      uint32_t MaxVectorWidth>
            struct OrthogonalCheck<
                MatrixLayout::
                    Col<BlockDim, BlockK, DataT, col_major, LhsVectorWidth, MaxVectorWidth>,
                MatrixLayout::
                    Row<BlockDim, BlockK, DataT, row_major, RhsVectorWidth, MaxVectorWidth>>
                : public std::true_type
            {
            };

        } // namespace detail

        template <typename DataLayoutT>
        using OrthogonalLayout_t = typename detail::OrthogonalLayout<DataLayoutT>::Type;

        // TODO: C++17 OrthogonalCheck_v
        // TODO: C++17 ConsistencyCheck_v

    } // namespace MatrixLayout

    namespace MatrixLayout
    {

        namespace detail
        {
            /* Pattern that maps threads contiguously to matrix columns and assumes
            * that VW will be mapped orthogonally to the column.
            * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
            *
            * Iterative thread stride cycles (same for all threads):
            *   Fill MaxVW => Fill BlockK => Fill BlockDim
            *
            * Example:
            *  BlockDim = 128   BlockK = 16
            *  MaxVW = 4       VW = 1
            *
            *  BlockDim Stride Count = 2, BlockDimStride = (64, 0)
            *  BlockK   Stride Count = 4, BlockKStride   = (0,  4)
            *  VW       Stride Count = 4, VWStride       = (0,  1)
            *
            *  Stride mapping (BlockDim, BlockK, VW)
            *  C_n = Matrix column
            *  i_n = cumulative iteration
            *
            *   kDim --------->
            *                     VW Stride
            *   BlockDim          |--1--|
            *   |                 |-- BlockK Stride = 4 --|
            *   |                 i0(0,0,0)   i2(0,0,2)   i4(0,1,0)   i6(0,1,2)         i14(0,3,2)
            *   |            --   v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
            *   v            |    |     |     |     |     |     |     |     |     |     |     ||     |
            *                |    |     |     |     |     |     |     |     |     |     |     ||     |
            *       BlockDim 64   | C0  |  C1 |  C2 |  C3 |  C4 |  C5 | C6  | C7  | ... | C14 || C15 |
            *        Stride  |    |     |     |     |     |     |     |     |     |     |     ||     |
            *                --   |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
            *                     i16(1,0,0)  i18(1,0,2)  i20(1,1,0)  i22(1,1,2)        i30(1,3,2)
            *                     v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
            *                     |     |     |     |     |     |     |     |     |     |     ||     |
            *                     |     |     |     |     |     |     |     |     |     |     ||     |
            *                     | C0  |  C1 |  C2 |  C3 | C4  | C5  | C6  | C7  | ... | C14 || C15 |
            *                     |     |     |     |     |     |     |     |     |     |     ||     |
            *                     |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
            *                     ^(128, 0)                                                           ^(BlockDim, BlockK)
            *   ...                                          ...
            *
            * Register file (for all VectorWidths = [1, MaxVectorWidth]):
            *
            * Elements 0..............63
            *           ______________
            *  Reg0    |  C0 [63:0]    |
            *  Reg1    |  C1 [63:0]    |
            *  Reg2    |  C2 [63:0]    |
            *  ...       ...
            *  Reg15   |  C15[63:0]    |
            *  Reg16   |  C0 [127:64]  |
            *  ...       ...
            *  Reg31   |  C15 [127:64] |
            }*/

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct ColOrthoVW
            {
                using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
                struct Traits
                {
                    enum : uint32_t
                    {
                        // Number of threads per wave
                        WaveSize = IOTraits::ThreadsPerIO,

                        // Number of BlockDim columns gathered per cycle of MaxVW
                        MaxKPerIO = WaveSize * MaxVectorWidth / std::min(BlockDim, WaveSize),

                        BlockDimStride_X = WaveSize,
                        BlockDimStride_Y = 0u,

                        BlockKStride_X = 0u,
                        BlockKStride_Y = MaxKPerIO,

                        VWStride_X = 0u,
                        VWStride_Y = VectorWidth,

                        // Flag for large BlockDim
                        LargeDim = BlockDim >= WaveSize,

                        // Number of segments in BlockDim direction
                        BlockDimSegs = std::max(BlockDim / BlockDimStride_X, 1u),

                        // Number of segments in the BlockK direction
                        BlockKSegs = BlockK / BlockKStride_Y,

                        // Number of segments in the MaxVW direction
                        VWSegs = MaxVectorWidth / VWStride_Y,

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

                    using MatrixCoordT = Coord2d;
                };

                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
                {
                    // TODO: Use constexpr if on C++17
                    if constexpr(Traits::LargeDim)
                    {
                        return make_coord2d(threadIdx.x % Traits::WaveSize, 0u);
                    }
                    else
                    {
                        return make_coord2d(threadIdx.x % BlockDim,
                                            (threadIdx.x / BlockDim) * MaxVectorWidth
                                                % Traits::MaxKPerIO);
                    }
                }

                ROCWMMA_DEVICE constexpr static inline auto strideCounts()
                {
                    return std::make_tuple((uint32_t)Traits::BlockDimSegs, // BlockDim Segments
                                           (uint32_t)Traits::BlockKSegs, // BlockK Segments
                                           (uint32_t)Traits::VWSegs); // VW Segments
                }

                ROCWMMA_DEVICE constexpr static inline auto strides()
                {
                    return std::make_tuple(
                        make_coord2d((uint32_t)Traits::BlockDimStride_X,
                                     (uint32_t)Traits::BlockDimStride_Y),
                        make_coord2d((uint32_t)Traits::BlockKStride_X,
                                     (uint32_t)Traits::BlockKStride_Y),
                        make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
                }

                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    incrementalOffset(uint32_t iteration)
                {
                    // TODO: Use constexpr if on C++ 17
                    if(Traits::LargeDim)
                    {
                        // incOffsetX:
                        // Minor cycle (VWSegs): = (iteration + 1) % VWSegs ? 0 : Wave size
                        // Major cycle (VWSegs * BlockDim):
                        // = (iteration + 1) % (VWSegs * BlockDimSegs) ? 0 : -BlockDim
                        constexpr int32_t IncXMinorStep = Traits::WaveSize;
                        constexpr int32_t IncXMajorStep = -BlockDim;

                        // incOffsetY:
                        // Minor cycle (VWSegs): = (iteration + 1) % VWSegs ? VW : -MaxVW
                        // Major cycle (VWSegs * BlockDim):
                        // = (iteration + 1) % (VWSegs * BlockDimSegs) ? MinorCycle : VW
                        constexpr int32_t IncYMinorStep = VectorWidth;
                        constexpr int32_t IncYMajorStep = -MaxVectorWidth;

                        // Bit masking for modulus operation
                        constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                        constexpr int32_t TotalSegsModMask
                            = LsbMask<Traits::Log2VWSegs + Traits::Log2BlockDimSegs>::value;

                        // Any remainder bits detected, mask = 0x0
                        // No remainder bits detected, mask = 0xFFFFFFFF
                        int32_t minorStepMask
                            = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                        int32_t majorStepMask
                            = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

                        return make_coord2d(
                            (IncXMinorStep & minorStepMask) + (majorStepMask & IncXMajorStep),
                            IncYMinorStep + ((minorStepMask ^ majorStepMask) & IncYMajorStep));
                    }
                    else
                    {
                        // incOffsetX: 0
                        // incOffsetY:
                        // Minor cycle (Every iteration): = VW
                        // Major cycle (VWSegs): = (iteration + 1) % VWSegs ? 0 : MaxVW * (WaveSegs - 1)
                        constexpr int32_t IncYMinorStep = VectorWidth;
                        constexpr int32_t IncYMajorStep = MaxVectorWidth * (Traits::WaveSegs - 1);
                        constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;

                        // Any remainder bits detected, mask = 0x0
                        // No remainder bits detected, mask = 0xFFFFFFFF
                        int32_t majorStepMask
                            = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                        return make_coord2d(0u, IncYMinorStep + (majorStepMask & IncYMajorStep));
                    }
                }
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    cumulativeOffset(uint32_t iteration)
                {
                    // TODO: Use constexpr if on C++17
                    if(Traits::LargeDim)
                    {
                        return make_coord2d(
                            iteration / Traits::VWSegs % Traits::BlockDimSegs * Traits::WaveSize,
                            iteration / (Traits::VWSegs * Traits::BlockDimSegs) * MaxVectorWidth
                                + iteration % Traits::VWSegs * VectorWidth);
                    }
                    else
                    {
                        return make_coord2d(0u,
                                            iteration / Traits::VWSegs
                                                    * (MaxVectorWidth * Traits::WaveSegs)
                                                + iteration % Traits::VWSegs * VectorWidth);
                    }
                }
            };

            /* Pattern that maps threads to matrix columns and assumes
            * that VW will be mapped inline with the column.
            * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
            *
            * Iterative thread stride cycles (same for all threads):
            *   Fill MaxVW => Fill BlockK => Fill BlockDim
            *
            * Example:
            * BlockDim = 256   BlockK = 4
            * MaxVW = 2       VW = 1
            *
            * BlockDim Stride Count = 4, BlockDimStride = (64, 0)
            * BlockK   Stride Count = 4, BlockKStride   = (0,  1)
            * VW       Stride Count = 2, VWStride       = (1,  0)
            *
            * Stride mapping (BlockDim, BlockK, VW)
            *  C_n = Matrix column
            *  i_n = cumulative iteration
            *
            *  iteration offsets:
            *  i0  = (0, 0)   i1  = (1, 0)  i2 = (0, 1) i3 = (1, 1)
            *  i4  = (0, 2)   i5  = (1, 2)  i6 = (0, 3) i7 = (1, 3)
            *  i8  = (0, 4)   i9  = (1, 4)  i10 = (0, 5) i11 = (1, 5)
            *  i12 = (0, 6)  i13  = (1, 6)  i14 = (0, 7) i15 = (1, 7)
            *
            *   kDim --------->
            *
            *   BlockDim1
            *   |                           |-- BlockK Stride = 4 --|
            *   |                           i0(0,0,0)   i4(0,2,0)
            *   |            _         _    v_____ _____v_____ _____
            *   v            |         |    |     |     |     |     |
            *                |  VW     1    |     |     |     |     |
            *       BlockDim |  Stride |    | C0  |  C1 |  C2 |  C3 |
            *        Stride  |         _    v     |     v     |     |
            *               64              i1(0,0,1)   i5(0,2,1)   |
            *                |              |     |     |     |     |
            *                |              |     |     |     |     |
            *                |              | C0  |  C1 |  C2 |  C3 |
            *                _              |_____|_____|_____|_____|
            *                               i8(1,0,0)   i12(1,2,0)
            *                               v_____ _____v_____ _____
            *                               |     |     |     |     |
            *                               |     |     |     |     |
            *                               | C0  |  C1 |  C2 |  C3 |
            *                               v     |     v     |     |
            *                               i9(1,0,1)   i13(1,2,1)  |
            *                               |     |     |     |     |
            *                               |     |     |     |     |
            *                               | C0  |  C1 |  C2 |  C3 |
            *                               |_____|_____|_____|_____|
            *                               ...                     ...
            *                               ...                     ...
            *                               ^(256, 0)               ^(BlockDim, BlockK)
            *
            * Register file (for all VectorWidths = [MaxVectorWidth, 1]):
            *
            * Elements 0...........1........................................... ............64
            *         ________________________________________________________________________
            * Reg0   |  C0E0   |  C0E2   | ... |  C0E62   |  C1E0   |  C1E2   | ... |  C1E62  |
            * Reg1   |  C0E1   |  C0E3   | ... |  C0E63   |  C1E1   |  C1E3   | ... |  C1E63  |
            * Reg2   |  C2E0   |  C2E2   | ... |  C2E62   |  C3E0   |  C3E2   | ... |  C3E62  |
            * Reg3   |  C2E1   |  C2E3   | ... |  C2E63   |  C3E1   |  C3E3   | ... |  C3E63  |
            * Reg4   |  C0E64  |  C0E66  | ... |  C0E126  |  C1E64  |  C1E66  | ... |  C1E126 |
            * Reg5   |  C0E65  |  C0E67  | ... |  C0E127  |  C1E65  |  C1E67  | ... |  C1E127 |
            * ...      ...
            * Reg10  |  C2E192 |  C2E194 | ... |  C2E254  |  C3E192 |  C3E194 | ... |  C3E254 |
            * Reg11  |  C2E193 |  C2E195 | ... |  C2E255  |  C3E193 |  C3E195 | ... |  C3E255 |
            *
            */

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct ColInlineVW
            {
                using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
                struct Traits
                {
                    enum : uint32_t
                    {
                        // Number of threads per wave
                        WaveSize = IOTraits::ThreadsPerIO,

                        // Number of elements per IO of MaxVW
                        MaxElementsPerIO = WaveSize * MaxVectorWidth,

                        // Number of BlockDim columns gathered per cycle of MaxVW
                        MaxKPerIO = std::max(1u, MaxElementsPerIO / BlockDim),

                        VWStride_X = VectorWidth,
                        VWStride_Y = 0u,

                        BlockDimStride_X = MaxElementsPerIO,
                        BlockDimStride_Y = 0u,

                        BlockKStride_X = 0u,
                        BlockKStride_Y = MaxKPerIO,

                        // Flag for large BlockDim
                        LargeDim = BlockDim >= MaxElementsPerIO,

                        // Number of segments in BlockDim direction
                        BlockDimSegs = std::max(1u, BlockDim / BlockDimStride_X),

                        // Number of segments in the BlockK direction
                        BlockKSegs = std::max(1u, BlockK / BlockKStride_Y),

                        // Number of segments in the MaxVW direction
                        VWSegs = std::max(1u, MaxVectorWidth / VWStride_X),

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

                    static_assert(BlockK >= MaxVectorWidth,
                                  "BlockK must be at least MaxVectorWidth");
                    static_assert(BlockK % MaxVectorWidth == 0,
                                  "BlockK must be a multiple of MaxVectorWidth");

                    using MatrixCoordT = Coord2d;
                };

                ROCWMMA_DEVICE constexpr static inline auto strideCounts()
                {
                    return std::make_tuple((uint32_t)Traits::BlockDimSegs, // BlockDim Segments
                                           (uint32_t)Traits::BlockKSegs, // BlockK Segments
                                           (uint32_t)Traits::VWSegs); // VW Segments
                }

                ROCWMMA_DEVICE constexpr static inline auto strides()
                {
                    return std::make_tuple(
                        make_coord2d((uint32_t)Traits::BlockDimStride_X,
                                     (uint32_t)Traits::BlockDimStride_Y),
                        make_coord2d((uint32_t)Traits::BlockKStride_X,
                                     (uint32_t)Traits::BlockKStride_Y),
                        make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
                }

                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
                {
                    // TODO: Use constexpr if when C++ 17
                    if(Traits::LargeDim)
                    {
                        return make_coord2d(threadIdx.x * MaxVectorWidth % Traits::MaxElementsPerIO,
                                            0u);
                    }
                    else
                    {
                        return make_coord2d(threadIdx.x * MaxVectorWidth % BlockDim,
                                            threadIdx.x * MaxVectorWidth / BlockDim
                                                % Traits::MaxKPerIO);
                    }
                }

                // Incremental iteration offset
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    incrementalOffset(uint32_t iteration)
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
                        int32_t VWSegsStepMask
                            = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;
                        int32_t TotalSegsStepMask
                            = static_cast<bool>((iteration + 1) & TotalSegsModMask) - 1;

                        return make_coord2d(IncX0MinorStep - (VWSegsStepMask & IncX0MajorStep)
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
                        int32_t majorStepMask
                            = static_cast<bool>((iteration + 1) & VWSegsModMask) - 1;

                        // Reference calculation:
                        // Iterative offsetX = VW - ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxVW
                        // Iterative offsetY = ((iteration + 1) % (MaxVectorWidth / VectorWidth) == 0) * MaxKPerIO
                        return make_coord2d(IncXMinorStep - (majorStepMask & IncXMajorStep),
                                            majorStepMask & IncYMajorStep);
                    }
                }

                // Cumulative iteration offset
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    cumulativeOffset(uint32_t iteration)
                {
                    // TODO: Use constexpr if when C++ 17
                    if(Traits::LargeDim)
                    {
                        constexpr int32_t VWSegsModMask = LsbMask<Traits::Log2VWSegs>::value;
                        constexpr int32_t BlockDimSegsModMask
                            = LsbMask<Traits::Log2BlockDimSegs>::value;

                        // Cumulative offsetX = (iteration / VWSegs) % BlockDimSegs * MaxElementsPerIO +
                        //                      (iteration % VWSegs) * VW,
                        // Cumulative offsetY = iteration / TotalSegs;
                        return make_coord2d(
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
                        return make_coord2d((iteration & VWSegsModMask) << Traits::Log2VW,
                                            iteration >> Traits::Log2VWSegs
                                                             << Traits::Log2MaxKPerIO);
                    }
                }
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct RowInlineVW
            {
                // RowInlineVW is orthogonal to ColInlineVW, therefore we can use reversed coordinates
                struct Traits
                {
                    using OrthoLayout
                        = ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
                    using MatrixCoordT = Coord2d;
                };

                // Matrix coord offsets
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
                {
                    return swap(Traits::OrthoLayout::baseOffset());
                }

                ROCWMMA_DEVICE constexpr static inline auto strideCounts()
                {
                    return Traits::OrthoLayout::strideCounts();
                }

                ROCWMMA_DEVICE constexpr static inline auto strides()
                {
                    auto t = Traits::OrthoLayout::strides();
                    return std::make_tuple(
                        swap(std::get<0>(t)), swap(std::get<1>(t)), swap(std::get<2>(t)));
                }

                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    incrementalOffset(uint32_t iteration)
                {
                    return swap(Traits::OrthoLayout::incrementalOffset(iteration));
                }
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    cumulativeOffset(uint32_t iteration)
                {
                    return swap(Traits::OrthoLayout::cumulativeOffset(iteration));
                }
            };

            template <uint32_t BlockDim,
                      uint32_t BlockK,
                      typename DataT,
                      uint32_t VectorWidth,
                      uint32_t MaxVectorWidth>
            struct RowOrthoVW
            {
                // RowOrthoVW is orthogonal to ColOrthoVW, therefore we can use reversed coordinates
                struct Traits
                {
                    using OrthoLayout
                        = ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;

                    using MatrixCoordT = Coord2d;
                };

                // Matrix coord offsets
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
                {
                    return swap(Traits::OrthoLayout::baseOffset());
                }

                ROCWMMA_DEVICE constexpr static inline auto strideCounts()
                {
                    return Traits::OrthoLayout::strideCounts();
                }

                ROCWMMA_DEVICE constexpr static inline auto strides()
                {
                    auto t = Traits::OrthoLayout::strides();
                    return std::make_tuple(
                        swap(std::get<0>(t)), swap(std::get<1>(t)), swap(std::get<2>(t)));
                }

                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    incrementalOffset(uint32_t iteration)
                {
                    return swap(Traits::OrthoLayout::incrementalOffset(iteration));
                }
                ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                    cumulativeOffset(uint32_t iteration)
                {
                    return swap(Traits::OrthoLayout::cumulativeOffset(iteration));
                }
            };

        } // namespace detail

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_IMPL_HPP
