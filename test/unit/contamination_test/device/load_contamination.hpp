/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2024 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_DEVICE_LOAD_CONTAMINATION_HPP
#define ROCWMMA_DEVICE_LOAD_CONTAMINATION_HPP

#include <rocwmma/internal/mapping_util.hpp>
#include <rocwmma/rocwmma.hpp>

#include "unit_test_traits.hpp"

namespace rocwmma
{

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  FragSize_guard<BlockM,
                                 BlockN,
                                 DataT,
                                 DataLayout,
                                 Constants::AMDGCN_WAVE_SIZE,
                                 Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationA(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;
        // Mapping:
        // Incoming -> Matrix A (ColNT)
        // BlockM -> BlockM
        // <Dummy> -> BlockN
        // BlockN -> BlockK
        auto frag = fragment<matrix_a, BlockM, 1, BlockN, DataT, DataLayout>();

        // Input is padded.
        // Make sure to offset read coords and extend reading ld.
        uint32_t paddedLd = ld
                            + 2
                                  * static_cast<uint32_t>(
                                      std::is_same<DataLayout, row_major>::value ? param2 : param1);
        auto readMatCoord = Mapping::matrixCoord();
        auto readMatCoordPadded
            = make_coord2d(get<0>(readMatCoord) + static_cast<uint32_t>(param1),
                           get<1>(readMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, readMatCoordPadded, paddedLd);
        auto* write = Mapping::dataCoord(out, ld);
        load_matrix_sync(frag, read, paddedLd);
        store_matrix_sync(write, frag, ld);
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  !FragSize_guard<BlockM,
                                  BlockN,
                                  DataT,
                                  DataLayout,
                                  Constants::AMDGCN_WAVE_SIZE,
                                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationA(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  FragSize_guard<BlockM,
                                 BlockN,
                                 DataT,
                                 DataLayout,
                                 Constants::AMDGCN_WAVE_SIZE,
                                 Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationB(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

        // Mapping:
        // Incoming -> Matrix B (RowNT)
        // <Dummy> -> BlockM
        // BlockN -> BlockN
        // BlockM -> BlockK
        auto frag = fragment<matrix_b, 1, BlockN, BlockM, DataT, DataLayout>();

        // Input is padded.
        // Make sure to offset read coords and extend reading ld.
        uint32_t paddedLd = ld
                            + 2
                                  * static_cast<uint32_t>(
                                      std::is_same<DataLayout, row_major>::value ? param2 : param1);
        auto readMatCoord = Mapping::matrixCoord();
        auto readMatCoordPadded
            = make_coord2d(get<0>(readMatCoord) + static_cast<uint32_t>(param1),
                           get<1>(readMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, readMatCoordPadded, paddedLd);
        auto* write = Mapping::dataCoord(out, ld);
        load_matrix_sync(frag, read, paddedLd);
        store_matrix_sync(write, frag, ld);
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  !FragSize_guard<BlockM,
                                  BlockN,
                                  DataT,
                                  DataLayout,
                                  Constants::AMDGCN_WAVE_SIZE,
                                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationB(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  FragSize_guard<BlockM,
                                 BlockN,
                                 DataT,
                                 DataLayout,
                                 Constants::AMDGCN_WAVE_SIZE,
                                 Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationAcc(uint32_t     m,
                                         uint32_t     n,
                                         DataT const* in,
                                         DataT*       out,
                                         uint32_t     ld,
                                         DataT        param1,
                                         DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

        // Mapping:
        // Incoming -> Matrix C (Row4T)
        // BlockM -> BlockM
        // BlockN -> BlockN
        // <Dummy> -> BlockK
        auto frag = fragment<accumulator, BlockM, BlockN, 1, DataT, DataLayout>();

        // Input is padded.
        // Make sure to offset read coords and extend reading ld.
        uint32_t paddedLd = ld
                            + 2
                                  * static_cast<uint32_t>(
                                      std::is_same<DataLayout, row_major>::value ? param2 : param1);
        auto readMatCoord = Mapping::matrixCoord();
        auto readMatCoordPadded
            = make_coord2d(get<0>(readMatCoord) + static_cast<uint32_t>(param1),
                           get<1>(readMatCoord) + static_cast<uint32_t>(param2));
        // Map, load and store.
        auto* read  = Mapping::dataCoord(in, readMatCoordPadded, paddedLd);
        auto* write = Mapping::dataCoord(out, ld);
        load_matrix_sync(frag, read, paddedLd);
        store_matrix_sync(write, frag, ld);
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename std::enable_if_t<
                  !FragSize_guard<BlockM,
                                  BlockN,
                                  DataT,
                                  DataLayout,
                                  Constants::AMDGCN_WAVE_SIZE,
                                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void loadContaminationAcc(uint32_t     m,
                                         uint32_t     n,
                                         DataT const* in,
                                         DataT*       out,
                                         uint32_t     ld,
                                         DataT        param1,
                                         DataT        param2)
    {
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_LOAD_CONTAMINATION_HPP
