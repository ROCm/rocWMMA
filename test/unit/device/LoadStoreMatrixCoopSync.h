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

#ifndef WMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_H
#define WMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_H

#include <algorithm>

#include "MappingUtil.h"
#include "WMMA.h"
#include "WMMACoop.h"

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
__global__ void __launch_bounds__(256) LoadStoreMatrixCoopSyncA(
    uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld, DataT param1, DataT param2)
{
    // Mapping:
    // Incoming -> Matrix A (ColNT)
    // BlockM -> BlockM
    // <Dummy> -> BlockN
    // BlockN -> BlockK
    auto frag = rocwmma::fragment<rocwmma::matrix_a, BlockM, 1, BlockN, DataT, Layout>();

    using Mapping = rocwmma::MappingUtil<BlockM, BlockN, DataT, Layout>;

    auto workgroupDim      = Mapping::workgroupDim();
    auto waveCoord         = Mapping::waveCoord();
    auto currentBlockCoord = Mapping::blockCoord();

    // sharingDim (param1):
    // 0 = waves in same row
    // 1 = waves in same col

    // sharingIndex (param2):
    // 0 = row/col 0 waves will cooperate
    // 1 = row/col 1 waves will cooperate
    // ...
    auto getFirst  = [](typename Mapping::WaveCoordT const& coord) { return coord.first; };
    auto getSecond = [](typename Mapping::WaveCoordT const& coord) { return coord.second; };

    auto sharingDim   = (uint32_t)param1;
    auto shareElement = (sharingDim == 0 ? getFirst : getSecond);
    auto coopElement  = (sharingDim == 0 ? getSecond : getFirst);

    auto sharingIndex = std::min((uint32_t)param2, shareElement(workgroupDim) - 1);
    if(shareElement(waveCoord) == sharingIndex)
    {

        // Get the slice of work
        auto workIndex = coopElement(waveCoord);
        auto workCount = coopElement(workgroupDim);

        // Start at the first block in WG coverage
        auto startBlockCoord
            = std::make_pair(std::get<0>(currentBlockCoord) - std::get<0>(waveCoord),
                             std::get<1>(currentBlockCoord) - std::get<1>(waveCoord));

        // Do cooperative loads for all blocks covered by WG
        for(int i = 0; i < std::get<0>(workgroupDim); i++)
        {
            for(int j = 0; j < std::get<1>(workgroupDim); j++)
            {
                // Map, load and store.
                auto  blockCoord = std::make_pair(std::get<0>(startBlockCoord) + i,
                                                  std::get<1>(startBlockCoord) + j);
                auto* read       = Mapping::dataCoord(in, Mapping::matrixCoord(blockCoord), ld);
                auto* write      = Mapping::dataCoord(out, Mapping::matrixCoord(blockCoord), ld);
                rocwmma::load_matrix_coop_sync(frag, read, ld, workIndex, workCount);
                rocwmma::store_matrix_coop_sync(write, frag, ld, workIndex, workCount);
            }
        }
    }
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
__global__ void __launch_bounds__(256) LoadStoreMatrixCoopSyncB(
    uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld, DataT param1, DataT param2)
{
    // Mapping:
    // Incoming -> Matrix B (RowNT)
    // <Dummy> -> BlockM
    // BlockN -> BlockN
    // BlockM -> BlockK
    auto frag = rocwmma::fragment<rocwmma::matrix_b, 1, BlockN, BlockM, DataT, Layout>();

    using Mapping = rocwmma::MappingUtil<BlockM, BlockN, DataT, Layout>;

    auto workgroupDim      = Mapping::workgroupDim();
    auto waveCoord         = Mapping::waveCoord();
    auto currentBlockCoord = Mapping::blockCoord();

    // sharingDim (param1):
    // 0 = waves in same row
    // 1 = waves in same col

    // sharingIndex (param2):
    // 0 = row/col 0 waves will cooperate
    // 1 = row/col 1 waves will cooperate
    // ...
    auto getFirst  = [](typename Mapping::WaveCoordT const& coord) { return coord.first; };
    auto getSecond = [](typename Mapping::WaveCoordT const& coord) { return coord.second; };

    auto sharingDim   = (uint32_t)param1;
    auto shareElement = (sharingDim == 0 ? getFirst : getSecond);
    auto coopElement  = (sharingDim == 0 ? getSecond : getFirst);

    auto sharingIndex = std::min((uint32_t)param2, shareElement(workgroupDim) - 1);
    if(shareElement(waveCoord) == sharingIndex)
    {

        // Get the slice of work
        auto workIndex = coopElement(waveCoord);
        auto workCount = coopElement(workgroupDim);

        // Start at the first block in WG coverage
        auto startBlockCoord
            = std::make_pair(std::get<0>(currentBlockCoord) - std::get<0>(waveCoord),
                             std::get<1>(currentBlockCoord) - std::get<1>(waveCoord));

        // Do cooperative loads for all blocks covered by WG
        for(int i = 0; i < std::get<0>(workgroupDim); i++)
        {
            for(int j = 0; j < std::get<1>(workgroupDim); j++)
            {
                // Map, load and store.
                auto  blockCoord = std::make_pair(std::get<0>(startBlockCoord) + i,
                                                  std::get<1>(startBlockCoord) + j);
                auto* read       = Mapping::dataCoord(in, Mapping::matrixCoord(blockCoord), ld);
                auto* write      = Mapping::dataCoord(out, Mapping::matrixCoord(blockCoord), ld);
                rocwmma::load_matrix_coop_sync(frag, read, ld, workIndex, workCount);
                rocwmma::store_matrix_coop_sync(write, frag, ld, workIndex, workCount);
            }
        }
    }
}

template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
__global__ void __launch_bounds__(256) LoadStoreMatrixCoopSyncAcc(
    uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld, DataT param1, DataT param2)
{
    // Mapping:
    // Incoming -> Matrix C (Row4T)
    // BlockM -> BlockM
    // BlockN -> BlockN
    // <Dummy> -> BlockK
    auto frag = rocwmma::fragment<rocwmma::accumulator, BlockM, BlockN, 1, DataT, Layout>();

    using Mapping = rocwmma::MappingUtil<BlockM, BlockN, DataT, Layout>;

    auto workgroupDim      = Mapping::workgroupDim();
    auto waveCoord         = Mapping::waveCoord();
    auto currentBlockCoord = Mapping::blockCoord();

    // sharingDim (param1):
    // 0 = waves in same row
    // 1 = waves in same col

    // sharingIndex (param2):
    // 0 = row/col 0 waves will cooperate
    // 1 = row/col 1 waves will cooperate
    // ...
    auto getFirst  = [](typename Mapping::WaveCoordT const& coord) { return coord.first; };
    auto getSecond = [](typename Mapping::WaveCoordT const& coord) { return coord.second; };

    auto sharingDim   = (uint32_t)param1;
    auto shareElement = (sharingDim == 0 ? getFirst : getSecond);
    auto coopElement  = (sharingDim == 0 ? getSecond : getFirst);

    auto sharingIndex = std::min((uint32_t)param2, shareElement(workgroupDim) - 1);
    if(shareElement(waveCoord) == sharingIndex)
    {
        // Get the slice of work
        auto workIndex = coopElement(waveCoord);
        auto workCount = coopElement(workgroupDim);

        // Start at the first block in WG coverage
        auto startBlockCoord
            = std::make_pair(std::get<0>(currentBlockCoord) - std::get<0>(waveCoord),
                             std::get<1>(currentBlockCoord) - std::get<1>(waveCoord));

        // Do cooperative loads for all blocks covered by WG
        for(int i = 0; i < std::get<0>(workgroupDim); i++)
        {
            for(int j = 0; j < std::get<1>(workgroupDim); j++)
            {
                // Map, load and store.
                auto  blockCoord = std::make_pair(std::get<0>(startBlockCoord) + i,
                                                  std::get<1>(startBlockCoord) + j);
                auto* read       = Mapping::dataCoord(in, Mapping::matrixCoord(blockCoord), ld);
                auto* write      = Mapping::dataCoord(out, Mapping::matrixCoord(blockCoord), ld);
                rocwmma::load_matrix_coop_sync(frag, read, ld, workIndex, workCount);
                rocwmma::store_matrix_coop_sync(write, frag, ld, workIndex, workCount);
            }
        }
    }
}

#endif // WMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_H
