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

#ifndef ROCWMMA_DEVICE_ROWNT_LAYOUT_H
#define ROCWMMA_DEVICE_ROWNT_LAYOUT_H

#include <rocwmma/internal/IOTraits.h>
#include <rocwmma/internal/Layout.h>
#include <rocwmma/internal/MappingUtil.h>

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename LayoutP>
    __global__ void RowNTLayout(uint32_t     m,
                                uint32_t     n,
                                DataT const* in,
                                DataT*       out,
                                uint32_t     ld,
                                DataT        param1,
                                DataT        param2)
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockM,

            MaxVectorWidth = detail::VecWidthTraits<BlockDim, KDim, DataT>::MaxVectorWidth,
            VectorWidth    = std::is_same<LayoutP, row_major>::value ? MaxVectorWidth : 1
        };

        using IOTraits = IOTraits<BlockDim, KDim, DataT, VectorWidth>;
        using LayoutT
            = MatrixLayout::RowNT<BlockDim, KDim, DataT, LayoutP, VectorWidth, MaxVectorWidth>;
        using Mapping = MappingUtil<BlockHeight, BlockWidth, DataT, LayoutP>;

        auto baseOffset  = LayoutT::baseOffset();
        auto iocount     = IOTraits::IOCount;
        auto matrixCoord = Mapping::matrixCoord();

        enum : uint32_t
        {
            MajorIndex = std::is_same<LayoutP, row_major>::value ? 0 : 1,
            MinorIndex = std::is_same<LayoutP, row_major>::value ? 1 : 0
        };

        for(uint32_t i = 0; i < iocount; ++i)
        {
            for(uint32_t j = 0; j < VectorWidth; j++)
            {
                auto index
                    = (std::get<MajorIndex>(matrixCoord) + std::get<MajorIndex>(baseOffset) + j)
                          * ld
                      + (std::get<MinorIndex>(matrixCoord) + std::get<MinorIndex>(baseOffset));
                out[index] = in[index];
            }
            baseOffset += LayoutT::incrementalOffset(i);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_ROWNT_LAYOUT_H
