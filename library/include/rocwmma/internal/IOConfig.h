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
#ifndef ROCWMMA_IO_CONFIG_H
#define ROCWMMA_IO_CONFIG_H

#include "Broadcast.h"
#include "Constants.h"
#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOTraits.h"
#include "Layout.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Pack.h"
#include "Types.h"
#include "Unpack.h"

namespace rocwmma
{
    namespace detail
    {
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout>
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
                  typename DataLayout>
        struct IOShape<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayout>
        {
            enum : uint32_t
            {
                BlockHeight = BlockM,
                BlockWidth  = BlockK,

                BlockDim = BlockM,
                KDim     = BlockK,

                MaxVectorWidth = VecWidthTraits<BlockDim, KDim, DataT>::MaxVectorWidth,
                VectorWidth
                = (std::is_same<DataLayout, row_major>::value && BlockDim < AMDGCN_WAVE_SIZE)
                      ? MaxVectorWidth
                      : 1
            };

            static_assert(!(std::is_same<DataLayout, col_major>::value && VectorWidth > 1),
                          "matrix_a in col_major currently does not support VectorWidth > 1");

            using DataMapper   = detail::DataSpace<DataLayout>;
            using MatrixMapper = MatrixLayout::
                ColNT<BlockDim, KDim, DataT, DataLayout, VectorWidth, MaxVectorWidth>;
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
                  typename DataLayout>
        struct IOShape<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayout>
        {
            enum : uint32_t
            {
                BlockHeight = BlockK,
                BlockWidth  = BlockN,

                BlockDim = BlockN,
                KDim     = BlockK,

                MaxVectorWidth = VecWidthTraits<BlockDim, KDim, DataT>::MaxVectorWidth,
                VectorWidth
                = (std::is_same<DataLayout, col_major>::value && BlockDim < AMDGCN_WAVE_SIZE)
                      ? MaxVectorWidth
                      : 1
            };

            static_assert(!(std::is_same<DataLayout, row_major>::value && VectorWidth > 1),
                          "matrix_b in row_major currently does not support VectorWidth > 1");

            using DataMapper   = detail::DataSpace<DataLayout>;
            using MatrixMapper = MatrixLayout::
                RowNT<BlockDim, KDim, DataT, DataLayout, VectorWidth, MaxVectorWidth>;
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
                  typename DataLayout>
        struct IOShape<accumulator, BlockM, BlockN, BlockK, DataT, DataLayout>
        {
            enum : uint32_t
            {
                BlockHeight = BlockM,
                BlockWidth  = BlockN,

                BlockDim = BlockN,
                KDim     = BlockM,

                MaxVectorWidth = std::is_same<DataT, float64_t>::value
                                     ? 1
                                     : 4, // Actual output of the mfma hardware
                VectorWidth    = std::is_same<DataLayout, col_major>::value ? MaxVectorWidth : 1,
            };

            static_assert(!(std::is_same<DataLayout, row_major>::value && VectorWidth > 1),
                          "accumulator in row_major currently does not support VectorWidth > 1");

            using DataMapper   = detail::DataSpace<DataLayout>;
            using MatrixMapper = MatrixLayout::
                RowNT<BlockDim, KDim, DataT, DataLayout, VectorWidth, MaxVectorWidth>;
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

            // No DataMapper or MatrixMapper without VW, MaxVW and DataOrientation info
        };

    } // namespace detail

    /**
 * \ingroup rocwmma
 * \defgroup ROCWMMA IOConfig
 *
 * @brief ROCWMMA fragment input and output configurations leveraging amdgcn architecture
 */

    /**
 * \ingroup ROCWMMA IOConfig
 * @{
 */

    /*! \class IOConfig
 *  \brief Definition of ROCWMMA fragment input / output configurations
 *         in specific matrix context.
 *
 * @tparam MatrixT - fragment context
 * @tparam BlockM/N/K - block dimensions
 * @tparam DataT - data type
 * @tparam DataLayout - in-memory layout as col_major or row_major
 *
 * BlockDim - leading block dimension (row / col size)
 * KDim - minor block dimension (row / col count)
 * MaxVectorWidth - maximum allowable vector width
 * VectorWidth - currently used vector width
 * CoopIndex - shared wave index (0 = row, 1 = col)
 * IOTraits - Input/output traits specific to AMDGCN architecture
 * Packer - Packs raw fragment data into register
 * Unpacker - Unpacks registers to raw fragment data
 * Broadcaster - Sets all fragment data to a desired value
 * MatrixLayout - Maps GPU threads to matrix shape or geometry
 * Loader - Issues load instructions for raw fragment data
 * Storer - Issues store instructions for raw fragment data
 * CoopLoader - Issues cooperative load instructions for raw fragment data
 * CoopStorer - Issues cooperative store instructions for raw fragment data
 */

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    struct IOConfig
    {
        using IOShape     = detail::IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;
        using IOTraits    = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT, IOShape::VectorWidth>;
        using Packer      = Pack<DataT, IOTraits::UnpackedSize>;
        using Unpacker    = Unpack<DataT, IOTraits::PackedSize>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;

        using MappingUtil
            = MappingUtil<IOShape::BlockHeight, IOShape::BlockWidth, DataT, DataLayout>;

        using Loader = OpaqueLoad<IOShape::BlockDim,
                                  IOShape::KDim,
                                  DataT,
                                  typename IOShape::DataMapper,
                                  typename IOShape::MatrixMapper,
                                  IOShape::VectorWidth>;

        using Storer = OpaqueStore<IOShape::BlockDim,
                                   IOShape::KDim,
                                   DataT,
                                   typename IOShape::DataMapper,
                                   typename IOShape::MatrixMapper,
                                   IOShape::VectorWidth>;

        using CoopLoader = CooperativeLoad<IOShape::BlockDim,
                                           IOShape::KDim,
                                           DataT,
                                           typename IOShape::DataMapper,
                                           typename IOShape::MatrixMapper,
                                           IOShape::VectorWidth>;

        using CoopStorer = CooperativeStore<IOShape::BlockDim,
                                            IOShape::KDim,
                                            DataT,
                                            typename IOShape::DataMapper,
                                            typename IOShape::MatrixMapper,
                                            IOShape::VectorWidth>;
    };

    /************************************************
 * Matrix C/D (accumulator) with undetermined DataLayout
 *
 * Fewer specific indications for matrix data geometry I/O, however
 * general IOTraits, Pack/Unpack, Broadcast still available.
 *
 * */
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    struct IOConfig<accumulator, BlockM, BlockN, BlockK, DataT, void>
    {
        using IOShape     = detail::IOShape<accumulator, BlockM, BlockN, BlockK, DataT, void>;
        using IOTraits    = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT>;
        using Packer      = Pack<DataT, IOTraits::UnpackedSize>;
        using Unpacker    = Unpack<DataT, IOTraits::PackedSize>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_CONFIG_H
