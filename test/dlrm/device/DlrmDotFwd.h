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

#ifndef DLRM_DOT_FWD_H
#define DLRM_DOT_FWD_H

#include "./Common.h"
#include "Utils.h"

template <typename DataT, uint TILE_DIM>
__global__ void __launch_bounds__(128, 1) dlrmDotFwd(const DataT* __restrict input,
                                                     DataT* __restrict output,
                                                     float32_t* acc,
                                                     uint       m,
                                                     uint       k,
                                                     uint       b,
                                                     uint       inputBatchOffset,
                                                     uint       outputBatchOffset,
                                                     uint       accBatchOffset)
{
    using MappingA   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
    using MappingB   = MappingUtil<TILE_DIM, TILE_DIM, DataT, col_major>;
    using MappingC   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
    using MappingAcc = MappingUtil<TILE_DIM, TILE_DIM, float32_t, row_major>;

    using FragA   = wmma::fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
    using FragB   = wmma::fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, DataT, col_major>;
    using FragAcc = wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float32_t>;

    // Copy bottom MLP to output
    // Threads with a global index < k are responsible for copying MLP data
    auto globalThreadCoord = blockIdx.x * blockDim.x + threadIdx.x;
    auto count             = k / blockDim.x;
    count                  = (count > 1) ? count : 1;
    if(blockIdx.x == 0 && blockIdx.y == 0)
    {
        for(int i = 0; i < count; i++)
        {
            if(i * blockDim.x + globalThreadCoord < k)
            {
                output[outputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord]
                    = input[inputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord];
            }
        }
    }

    // Target output block
    auto matrixCoordC = MappingC::matrixCoord();

    if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < m)
    {
        // Initialize accumulator
        auto fragAcc = FragAcc();
        wmma::fill_fragment(fragAcc, static_cast<float32_t>(0));

        // Setup starting addresses
        auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
        auto* addrA
            = MappingA::dataCoord(inputWithOffset, k, std::make_pair(std::get<0>(matrixCoordC), 0));
        auto* addrB
            = MappingB::dataCoord(inputWithOffset, k, std::make_pair(0, std::get<1>(matrixCoordC)));

        // Setup address increments.
        // A steps BlockK through m x k
        // B steps BlockK through k x m
        auto incrA = MappingA::dataOffset(k, std::make_pair(0, TILE_DIM));
        auto incrB = MappingB::dataOffset(k, std::make_pair(TILE_DIM, 0));

        auto count = k / TILE_DIM;
        for(int i = 0; i < count; i++)
        {
            auto fragA = FragA();
            auto fragB = FragB();

            // Load and multiply
            wmma::load_matrix_sync(fragA, addrA, k);
            wmma::load_matrix_sync(fragB, addrB, k);
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

            addrA += incrA;
            addrB += incrB;
        }

        // Store fragAcc to global acc
        auto* accWithOffset = acc + accBatchOffset * blockIdx.z;
        auto* addrAcc       = MappingAcc::dataCoord(accWithOffset, m, matrixCoordC);
        wmma::store_matrix_sync(addrAcc, fragAcc, m, wmma::mem_row_major);

        // Copy lower triangular from acc to output
        auto fragColIdx   = threadIdx.x % TILE_DIM;
        auto globalColIdx = std::get<1>(matrixCoordC) + fragColIdx;
        auto rowsPerStep  = AMDGCN_WAVE_SIZE / TILE_DIM;

        count = (TILE_DIM * TILE_DIM) >> Log2<AMDGCN_WAVE_SIZE>::value;
        for(int i = 0; i < count; i++)
        {
            auto fragRowIdx = i * rowsPerStep + ((threadIdx.x & (AMDGCN_WAVE_SIZE - 1)) / TILE_DIM);
            auto globalRowIdx = std::get<0>(matrixCoordC) + fragRowIdx;
            if(globalRowIdx > globalColIdx)
            {
                auto outputOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);
                output[outputBatchOffset * blockIdx.z + outputOffset + globalColIdx]
                    = static_cast<DataT>(
                        acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx]);
            }
        }
    }
}

#endif // DLRM_DOT_FWD_H
