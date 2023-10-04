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

#ifndef DLRM_DOT_BWD_LDS_HPP
#define DLRM_DOT_BWD_LDS_HPP

#include "./common.hpp"
#include "./lds_mapping_util.hpp"

namespace rocwmma
{

    template <typename DataT>
    __global__ void trilReconstructLds(const DataT* __restrict upstreamGrad,
                                       DataT* __restrict acc,
                                       uint m,
                                       uint k,
                                       uint b,
                                       uint upstreamBatchOffset,
                                       uint accBatchOffset)
    {
        auto blocksPerRow = (m + blockDim.x - 1) / blockDim.x;
        int  globalRowIdx;
        if(blockDim.x >= m)
        {
            globalRowIdx = blockIdx.x * (blockDim.x / m) + (threadIdx.x / m);
        }
        else
        {
            globalRowIdx = blockIdx.x / blocksPerRow;
        }

        auto globalColIdx = (blockIdx.x * blockDim.x + threadIdx.x) % m;
        if(globalRowIdx < m && globalColIdx < m)
        {
            if(globalRowIdx == globalColIdx)
            {
                acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx] = 0.0;
            }
            else if(globalRowIdx > globalColIdx)
            {
                auto upstreamGradOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);

                // original tril copy
                acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx]
                    = upstreamGrad[upstreamBatchOffset * blockIdx.z + upstreamGradOffset
                                   + globalColIdx];

                // transposed tril copy
                acc[accBatchOffset * blockIdx.z + globalColIdx * m + globalRowIdx]
                    = upstreamGrad[upstreamBatchOffset * blockIdx.z + upstreamGradOffset
                                   + globalColIdx];
            }
        }
    }

    template <typename DataT, uint TILE_DIM, typename LdsMapping>
    __global__ void __launch_bounds__(128, 1) dlrmDotBwdLds(const DataT* __restrict input,
                                                            const DataT* __restrict upstreamGrad,
                                                            DataT* __restrict grad,
                                                            DataT* __restrict bottomMlpGrad,
                                                            DataT* __restrict acc,
                                                            uint m,
                                                            uint k,
                                                            uint b,
                                                            uint inputBatchOffset,
                                                            uint upstreamBatchOffset,
                                                            uint accBatchOffset)
    {
        using TileMapping = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;

        using FragA   = fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
        using FragB   = fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
        using FragC   = fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, DataT>;
        using FragAcc = fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float32_t>;

        // Will store to LDS as though it were a register file.
        // Rows = register count
        // Cols = unpacked register elements = 64
        // Row major to minimize bank conflicts
        using MappingLds = LdsMappingUtil<TILE_DIM,
                                          TILE_DIM,
                                          TILE_DIM,
                                          DataT,
                                          row_major,
                                          row_major,
                                          row_major,
                                          LdsMapping,
                                          1,
                                          1>;

        // Copy bottom MLP grad
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
                    bottomMlpGrad[k * blockIdx.z + i * blockDim.x + globalThreadCoord]
                        = upstreamGrad[upstreamBatchOffset * blockIdx.z + i * blockDim.x
                                       + globalThreadCoord];
                }
            }
        }

        // Target accumulator block
        auto matrixCoord = TileMapping::matrixCoord();

        // Target output gradient block to perform reverse bmm
        if(get<0>(matrixCoord) < m && get<1>(matrixCoord) < k)
        {
            // Initialize accumulator
            auto fragAcc = FragAcc();
            fill_fragment(fragAcc, static_cast<float32_t>(0));

            // Setup starting addresses
            auto* accWithOffset   = acc + accBatchOffset * blockIdx.z;
            auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
            auto* addrA
                = TileMapping::dataCoord(accWithOffset, make_coord2d(get<0>(matrixCoord), 0), m);
            auto* addrB
                = TileMapping::dataCoord(inputWithOffset, make_coord2d(0, get<1>(matrixCoord)), k);

            /// Setup LDS addressing and start writing pre-fetch to LDS
            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto* ldsPtrLo = reinterpret_cast<DataT*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + MappingLds::ldsWidth() * MappingLds::ldsHeight();

            // Prefetch the first block from global memory
            if(m / TILE_DIM > 1)
            {
                MappingLds::prefetchCoopGlobalA(ldsPtrLo, addrA, m);
                MappingLds::prefetchCoopGlobalB(ldsPtrLo, addrB, k);
            }
            else
            {
                MappingLds::prefetchGlobalA(ldsPtrLo, addrA, m);
                MappingLds::prefetchGlobalB(ldsPtrLo, addrB, k);
            }

            // Wait for A / B write LDS
            synchronize_workgroup();

            // Setup address increments.
            // A steps BlockK through m x m
            // B steps BlockK through m x k
            auto fragA = FragA();
            auto fragB = FragB();

            auto incrA = TileMapping::dataOffset(make_coord2d(0, TILE_DIM), m);
            auto incrB = TileMapping::dataOffset(make_coord2d(TILE_DIM, 0), k);

            auto endA = addrA + incrA * (m / TILE_DIM);

            addrA += incrA;
            addrB += incrB;

            while(addrA != endA)
            {
                // Cache lds blocks to register
                MappingLds::prefetchLocalA(fragA, ldsPtrLo, 0);
                MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);

                // Start pulling in the next block
                if(m / TILE_DIM > 1)
                {
                    MappingLds::prefetchCoopGlobalA(ldsPtrHi, addrA, m);
                    MappingLds::prefetchCoopGlobalB(ldsPtrHi, addrB, k);
                }
                else
                {
                    MappingLds::prefetchGlobalA(ldsPtrHi, addrA, m);
                    MappingLds::prefetchGlobalB(ldsPtrHi, addrB, k);
                }

                // Mma for current block
                mma_sync(fragAcc, fragA, fragB, fragAcc);

                // Wait for A / B read LDS
                synchronize_workgroup();

                addrA += incrA;
                addrB += incrB;

                auto* tmp = ldsPtrLo;
                ldsPtrLo  = ldsPtrHi;
                ldsPtrHi  = tmp;
            }

            // Mma for the last block
            MappingLds::prefetchLocalA(fragA, ldsPtrLo, 0);
            MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);
            mma_sync(fragAcc, fragA, fragB, fragAcc);

            // Wait for final mma before writing to LDS
            synchronize_workgroup();

            // Output address
            auto* gradWithOffset = grad + inputBatchOffset * blockIdx.z;
            auto* addrGrad       = TileMapping::dataCoord(gradWithOffset, matrixCoord, k);

            // Store accumulator fragment to output gradient
            auto fragC = FragC();

#pragma unroll
            for(int i = 0; i < fragC.num_elements; i++)
            {
                fragC.x[i] = static_cast<DataT>(fragAcc.x[i]);
            }

            // Store the output
            store_matrix_sync(addrGrad, fragC, k, mem_row_major);
        }
    }

} // namespace rocwmma

#endif // DLRM_DOT_BWD_LDS_HPP
