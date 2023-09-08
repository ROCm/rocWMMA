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
#ifndef ROCWMMA_IO_CONFIG_HPP
#define ROCWMMA_IO_CONFIG_HPP

#include "broadcast.hpp"
#include "coop_load.hpp"
#include "coop_store.hpp"
#include "io_shape.hpp"
#include "opaque_load.hpp"
#include "opaque_store.hpp"
#include "pack_util.hpp"
#include "types.hpp"

namespace rocwmma
{

    /**
     * \defgroup Rocwmma_ioconf ROCWMMA IOConfig
     * @brief ROCWMMA fragment input and output configurations leveraging amdgcn architecture
     * @{
     */

    /*! \struct IOConfig
 *  \brief Definition of ROCWMMA fragment input / output configurations
 *         in specific matrix context.
 *
 * @tparam Matrix fragment context
 * @tparam BlockM/N/K block dimensions
 * @tparam DataT data type
 * @tparam DataLayout in-memory layout as col_major or row_major
 * @param BlockDim leading block dimension (row / col size)
 * @param KDim minor block dimension (row / col count)
 * @param MaxVectorWidth maximum allowable vector width
 * @param VectorWidth currently used vector width
 * @param CoopIndex shared wave index (0 = row, 1 = col)
 * @param IOTraits Input/output traits specific to AMDGCN architecture
 * @param Packer Packs raw fragment data into register
 * @param Unpacker Unpacks registers to raw fragment data
 * @param Broadcaster Sets all fragment data to a desired value
 * @param MatrixLayout Maps GPU threads to matrix shape or geometry
 * @param Loader Issues load instructions for raw fragment data
 * @param Storer Issues store instructions for raw fragment data
 * @param CoopLoader Issues cooperative load instructions for raw fragment data
 * @param CoopStorer Issues cooperative store instructions for raw fragment data
 */

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOConfig
    {
        using IOShape     = IOShape<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>;
        using IOTraits    = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT, IOShape::VectorWidth>;
        using PackUtil    = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;

        using MappingUtil
            = MappingUtil<IOShape::BlockHeight, IOShape::BlockWidth, DataT, DataLayoutT>;

        using Loader = OpaqueLoad<IOShape::BlockDim,
                                  IOShape::KDim,
                                  DataT,
                                  typename IOShape::DataLayout,
                                  typename IOShape::MatrixLayout,
                                  IOShape::VectorWidth>;

        using Storer = OpaqueStore<IOShape::BlockDim,
                                   IOShape::KDim,
                                   DataT,
                                   typename IOShape::DataLayout,
                                   typename IOShape::MatrixLayout,
                                   IOShape::VectorWidth>;

        using CoopLoader = CooperativeLoad<IOShape::BlockDim,
                                           IOShape::KDim,
                                           DataT,
                                           typename IOShape::DataLayout,
                                           typename IOShape::MatrixLayout,
                                           IOShape::VectorWidth>;

        using CoopStorer = CooperativeStore<IOShape::BlockDim,
                                            IOShape::KDim,
                                            DataT,
                                            typename IOShape::DataLayout,
                                            typename IOShape::MatrixLayout,
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
        using IOShape     = IOShape<accumulator, BlockM, BlockN, BlockK, DataT, void>;
        using IOTraits    = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT>;
        using PackUtil    = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;
    };
    /** @}*/

} // namespace rocwmma

#endif // ROCWMMA_IO_CONFIG_HPP
