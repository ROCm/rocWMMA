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

#ifndef WMMA_DEVICE_MAP_THREAD_TO_MATRIX_H
#define WMMA_DEVICE_MAP_THREAD_TO_MATRIX_H

#include "MappingUtil.h"
#include "WMMA.h"

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    __global__ void MapThreadToMatrix(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, Layout>;

        enum : uint32_t
        {
            MajorIndex = std::is_same<Layout, row_major>::value ? 0 : 1,
            MinorIndex = std::is_same<Layout, row_major>::value ? 1 : 0,
            ldmajor    = std::is_same<Layout, row_major>::value ? BlockM : BlockN,
            ldminor    = std::is_same<Layout, row_major>::value ? BlockN : BlockM
        };

        auto majCoord = (std::is_same<Layout, row_major>::value
                             ? ((threadIdx.x + blockDim.x * blockIdx.x) / AMDGCN_WAVE_SIZE)
                             : (threadIdx.y + blockDim.y * blockIdx.y))
                        * ldmajor;
        auto minCoord = (std::is_same<Layout, row_major>::value
                             ? (threadIdx.y + blockDim.y * blockIdx.y)
                             : ((threadIdx.x + blockDim.x * blockIdx.x) / AMDGCN_WAVE_SIZE))
                        * ldminor;

        for(int i = 0; i < ldminor; ++i)
        {
            for(int j = 0; j < ldmajor; ++j)
            {
                out[(majCoord + j) * ld + (minCoord + i)]
                    = in[(((majCoord + j) * ld) + (minCoord + i))];
            }
        }
    }

} // namespace rocwmma

#endif // WMMA_DEVICE_MAP_THREAD_TO_MATRIX_H
