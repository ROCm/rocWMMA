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
#ifndef ROCWMMA_IO_LAYOUT_HPP
#define ROCWMMA_IO_LAYOUT_HPP

#include "constants.hpp"
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

    /*! \struct IOLayout
 *  \brief Definition of VW, MaxVW, data and matrix mapping utilities
 *         in specific matrix context.
 *
 * @tparam MatrixT fragment context
 * @tparam BlockDim Block leading dimension
 * @tparam BlockK Block K-dimension
 * @tparam DataT data type
 * @tparam DataLayoutT in-memory layout as col_major or row_major
 * @tparam WaveCount number of cooperative waves
 */
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
            VW = is_same<DataLayoutT, row_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template ColNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(is_same_v<DataLayoutT, col_major> && VW > 1),
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
            VW = is_same<DataLayoutT, col_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template RowNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(is_same_v<DataLayoutT, row_major> && VW > 1),
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
            MaxVW = (is_same<DataT, float64_t>::value || ROCWMMA_ARCH_GFX11) ? 1u : 4u,
            VW    = is_same<DataLayoutT, col_major>::value ? MaxVW : 1u
        };

        // Layout mapping for 1d / 2d
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;
        using MatrixLayout
            = MatrixLayout::template RowNT<BlockDim, KDim, DataT, DataLayoutT, VW, MaxVW>;

        static_assert(!(is_same<DataLayoutT, row_major>::value && VW > 1),
                      "accumulator in row_major currently does not support VW > 1");
    };

    template <uint32_t BlockDim, uint32_t KDim, typename DataT, uint32_t WaveCount>
    struct IOLayout<accumulator, BlockDim, KDim, DataT, void, WaveCount>
    {
        // No layout mapping without VW, MaxVW and DataLayoutT info
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_LAYOUT_HPP
