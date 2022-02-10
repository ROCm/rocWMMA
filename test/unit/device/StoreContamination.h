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

#ifndef ROCWMMA_DEVICE_STORE_CONTAMINATION_H
#define ROCWMMA_DEVICE_STORE_CONTAMINATION_H

#include <WMMA/WMMA.h>
#include <WMMA/internal/MappingUtil.h>

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    __global__ void storeContaminationA(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, Layout>;

        // Mapping:
        // Incoming -> Matrix A (ColNT)
        // BlockM -> BlockM
        // <Dummy> -> BlockN
        // BlockN -> BlockK
        auto frag = fragment<matrix_a, BlockM, 1, BlockN, DataT, Layout>();

        // Output is padded.
        // Make sure to offset write coords and extend writing ld.
        uint32_t paddedLd
            = ld
              + 2 * static_cast<uint32_t>(std::is_same<Layout, row_major>::value ? param2 : param1);
        auto writeMatCoord = Mapping::matrixCoord();
        auto writeMatCoordPadded
            = std::make_pair(std::get<0>(writeMatCoord) + static_cast<uint32_t>(param1),
                             std::get<1>(writeMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, ld);
        auto* write = Mapping::dataCoord(out, writeMatCoordPadded, paddedLd);
        load_matrix_sync(frag, read, ld);
        store_matrix_sync(write, frag, paddedLd);
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    __global__ void storeContaminationB(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, Layout>;

        // Mapping:
        // Incoming -> Matrix B (RowNT)
        // <Dummy> -> BlockM
        // BlockN -> BlockN
        // BlockM -> BlockK
        auto frag = fragment<matrix_b, 1, BlockN, BlockM, DataT, Layout>();

        // Output is padded.
        // Make sure to offset write coords and extend writing ld.
        uint32_t paddedLd
            = ld
              + 2 * static_cast<uint32_t>(std::is_same<Layout, row_major>::value ? param2 : param1);
        auto writeMatCoord = Mapping::matrixCoord();
        auto writeMatCoordPadded
            = std::make_pair(std::get<0>(writeMatCoord) + static_cast<uint32_t>(param1),
                             std::get<1>(writeMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, ld);
        auto* write = Mapping::dataCoord(out, writeMatCoordPadded, paddedLd);
        load_matrix_sync(frag, read, ld);
        store_matrix_sync(write, frag, paddedLd);
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    __global__ void storeContaminationAcc(uint32_t     m,
                                          uint32_t     n,
                                          DataT const* in,
                                          DataT*       out,
                                          uint32_t     ld,
                                          DataT        param1,
                                          DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, Layout>;

        // Mapping:
        // Incoming -> Matrix C (Row4T)
        // BlockM -> BlockM
        // BlockN -> BlockN
        // <Dummy> -> BlockK
        auto frag = fragment<accumulator, BlockM, BlockN, 1, DataT, Layout>();

        // Output is padded.
        // Make sure to offset write coords and extend writing ld.
        uint32_t paddedLd
            = ld
              + 2 * static_cast<uint32_t>(std::is_same<Layout, row_major>::value ? param2 : param1);
        auto writeMatCoord = Mapping::matrixCoord();
        auto writeMatCoordPadded
            = std::make_pair(std::get<0>(writeMatCoord) + static_cast<uint32_t>(param1),
                             std::get<1>(writeMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, ld);
        auto* write = Mapping::dataCoord(out, writeMatCoordPadded, paddedLd);
        load_matrix_sync(frag, read, ld);
        store_matrix_sync(write, frag, paddedLd);
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_STORE_CONTAMINATION_H
