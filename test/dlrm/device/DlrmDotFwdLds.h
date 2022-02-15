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

#ifndef DLRM_DOT_FWD_LDS_H
#define DLRM_DOT_FWD_LDS_H

#include "../../gemm/LdsMappingUtil.h"
#include "./Common.h"
#include "Utils.h"

namespace rocwmma
{

    template <typename DataT, uint TILE_DIM, typename LdsMapping>
    __global__ void __launch_bounds__(128, 1) dlrmDotFwdLds(const DataT* __restrict input,
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

        using FragA   = fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
        using FragB   = fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, DataT, col_major>;
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
                                          col_major,
                                          row_major,
                                          LdsMapping,
                                          1,
                                          1>;

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
            fill_fragment(fragAcc, static_cast<float32_t>(0));

            // Setup starting global addresses
            auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
            auto* addrA           = MappingA::dataCoord(
                          inputWithOffset, std::make_pair(std::get<0>(matrixCoordC), 0), k);
            auto* addrB = MappingB::dataCoord(
                inputWithOffset, std::make_pair(0, std::get<1>(matrixCoordC)), k);

            /// Setup LDS addressing and start writing pre-fetch to LDS
            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto* ldsPtrLo = reinterpret_cast<DataT*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + MappingLds::ldsWidth() * MappingLds::ldsHeight();

            // Prefetch the first block from global memory
            MappingLds::prefetchGlobalA(ldsPtrLo, addrA, k);
            MappingLds::prefetchGlobalB(ldsPtrLo, addrB, k);

            // Wait for A / B write LDS
            synchronize_workgroup();

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x m
            auto fragA = FragA();
            auto fragB = FragB();

            auto incrA = MappingA::dataOffset(std::make_pair(0, TILE_DIM), k);
            auto incrB = MappingB::dataOffset(std::make_pair(TILE_DIM, 0), k);

            auto endA = addrA + incrA * (k / TILE_DIM);

            addrA += incrA;
            addrB += incrB;
            while(addrA != endA)
            {
                // Cache lds blocks to register
                MappingLds::prefetchLocalA(fragA, ldsPtrLo, 0);
                fill_fragment(fragB, static_cast<DataT>(1));
                //MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);

                // Start pulling in the next block
                MappingLds::prefetchGlobalA(ldsPtrHi, addrA, k);
                MappingLds::prefetchGlobalB(ldsPtrHi, addrB, k);

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
            fill_fragment(fragB, static_cast<DataT>(1));
            // MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);
            mma_sync(fragAcc, fragA, fragB, fragAcc);

            //Store fragAcc to global acc
            auto* accWithOffset = acc + accBatchOffset * blockIdx.z;
            auto* addrAcc       = MappingAcc::dataCoord(accWithOffset, matrixCoordC, m);
            store_matrix_sync(addrAcc, fragAcc, m, mem_row_major);

            // Wait for final mma before writing to LDS
            synchronize_workgroup();

            // Store acc frag to lds for recasting
            auto* ldsPtrAcc     = reinterpret_cast<float32_t*>(localMemPtr);
            auto* ldsWavePtrAcc = ldsPtrAcc + MappingLds::waveOffsetA();
            store_matrix_sync(ldsWavePtrAcc, fragAcc, MappingLds::ld(), mem_row_major);

            // Wait for LDS write before accessing
            synchronize_workgroup();

            // Copy lower triangular from lds to output
            auto fragColIdx   = threadIdx.x % TILE_DIM;
            auto globalColIdx = std::get<1>(matrixCoordC) + fragColIdx;
            auto rowsPerStep  = AMDGCN_WAVE_SIZE / TILE_DIM;

            count = (TILE_DIM * TILE_DIM) >> Log2<AMDGCN_WAVE_SIZE>::value;
            for(int i = 0; i < count; i++)
            {
                auto fragRowIdx
                    = i * rowsPerStep + ((threadIdx.x & (AMDGCN_WAVE_SIZE - 1)) / TILE_DIM);
                auto globalRowIdx = std::get<0>(matrixCoordC) + fragRowIdx;
                if(globalRowIdx > globalColIdx)
                {
                    auto outputOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);
                    output[outputBatchOffset * blockIdx.z + outputOffset + globalColIdx]
                        = static_cast<DataT>(ldsWavePtrAcc[fragRowIdx * TILE_DIM + fragColIdx]);
                }
            }

            // // Copy lower triangular from acc to output
            // auto fragColIdx   = threadIdx.x % TILE_DIM;
            // auto globalColIdx = std::get<1>(matrixCoordC) + fragColIdx;
            // auto rowsPerStep  = AMDGCN_WAVE_SIZE / TILE_DIM;

            // count = (TILE_DIM * TILE_DIM) >> Log2<AMDGCN_WAVE_SIZE>::value;
            // for(int i = 0; i < count; i++)
            // {
            //     auto fragRowIdx
            //         = i * rowsPerStep + ((threadIdx.x & (AMDGCN_WAVE_SIZE - 1)) / TILE_DIM);
            //     auto globalRowIdx = std::get<0>(matrixCoordC) + fragRowIdx;
            //     if(globalRowIdx > globalColIdx)
            //     {
            //         auto outputOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);
            //         output[outputBatchOffset * blockIdx.z + outputOffset + globalColIdx]
            //             = static_cast<DataT>(
            //                 acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx]);
            //     }
            // }
        }
    }
} // namespace rocwmma

// template <typename DataT, uint TILE_DIM, typename LdsMapping>
//     __global__ void __launch_bounds__(128, 1) dlrmDotFwdLds(const DataT* __restrict input,
//                                                          DataT* __restrict output,
//                                                          float32_t* acc,
//                                                          uint       m,
//                                                          uint       k,
//                                                          uint       b,
//                                                          uint       inputBatchOffset,
//                                                          uint       outputBatchOffset,
//                                                          uint       accBatchOffset)
//     {
//         using MappingA   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
//         using MappingB   = MappingUtil<TILE_DIM, TILE_DIM, DataT, col_major>;
//         using MappingC   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
//         using MappingAcc = MappingUtil<TILE_DIM, TILE_DIM, float32_t, row_major>;

//         using FragA   = fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
//         using FragB   = fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, DataT, col_major>;
//         using FragAcc = fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float32_t>;

//         // Will store to LDS as though it were a register file.
//         // Rows = register count
//         // Cols = unpacked register elements = 64
//         // Row major to minimize bank conflicts
//         using MappingLdsA = MappingUtil<FragA::size(), AMDGCN_WAVE_SIZE, DataT, row_major>;
//         using MappingLdsB = MappingUtil<FragB::size(), AMDGCN_WAVE_SIZE, DataT, row_major>;
//         using FragLdsA
//             = fragment<register_file, 1, AMDGCN_WAVE_SIZE, FragA::size(), DataT, row_major>;
//         using FragLdsB
//             = fragment<register_file, 1, AMDGCN_WAVE_SIZE, FragB::size(), DataT, row_major>;

//         static_assert(FragA::size() * 64 == TILE_DIM * TILE_DIM, "Elements of A don't match");
//         static_assert(FragLdsA::size() == FragA::size(), "A Sizes don't match");
//         static_assert(FragB::size() * 64 == TILE_DIM * TILE_DIM, "Elements don't match");
//         static_assert(FragLdsB::size() == FragB::size(), "Sizes don't match");

//         // Copy bottom MLP to output
//         // Threads with a global index < k are responsible for copying MLP data
//         auto globalThreadCoord = blockIdx.x * blockDim.x + threadIdx.x;
//         auto count             = k / blockDim.x;
//         count                  = (count > 1) ? count : 1;
//         if(blockIdx.x == 0 && blockIdx.y == 0)
//         {
//             for(int i = 0; i < count; i++)
//             {
//                 if(i * blockDim.x + globalThreadCoord < k)
//                 {
//                     output[outputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord]
//                         = input[inputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord];
//                 }
//             }
//         }

//         // Target output block
//         auto matrixCoordC = MappingC::matrixCoord();

//         if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < m)
//         {
//             // Initialize accumulator
//             auto fragAcc = FragAcc();
//             fill_fragment(fragAcc, static_cast<float32_t>(0));

//             // Setup starting addresses
//             auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
//             auto* addrA           = MappingA::dataCoord(
//                           inputWithOffset, std::make_pair(std::get<0>(matrixCoordC), 0), k);
//             auto* addrB = MappingB::dataCoord(
//                 inputWithOffset, std::make_pair(0, std::get<1>(matrixCoordC)), k);

//             // Prefetch the first block from global memory
//             auto fragA = FragA();
//             auto fragB = FragB();
//             load_matrix_sync(fragA, addrA, k);
//             load_matrix_sync(fragB, addrB, k);

//             // Setup a register file in LDS which is friendly to minimizing bank conflicts.
//             // Register file blocks in LDS follow same wg mapping for convenience.
//             // Each wave will prefetch one block of A and one block of B.
//             // A blocks occupy first portion of LDS and B blocks occupy the latter.
//             HIP_DYNAMIC_SHARED(void*, localMemPtr);
//             auto workgroupDim = MappingLdsA::workgroupDim();
//             auto ldLds = AMDGCN_WAVE_SIZE * std::get<1>(workgroupDim);

//             auto* baseAddrLdsA = reinterpret_cast<DataT>(localMemPtr);
//             auto* baseAddrLdsB = baseAddrLdsA + std::get<0>(workgroupDim) * FragLdsA::size() * ldLds;

//             auto matrixCoordLdsA = MappingLdsA::matrixCoord(MappingLdsA::waveCoord());
//             auto matrixCoordLdsB = MappingLdsB::matrixCoord(MappingLdsB::waveCoord());

//             auto* addrLdsA = MappingLdsA::dataCoord(baseAddrLdsA, matrixCoordLdsA, ldLds);
//             auto* addrLdsB = MappingLdsA::dataCoord(baseAddrLdsB, matrixCoordLdsB, ldLds);

//             store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragA), ldLds);
//             store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragB), ldLds);

//             // Setup address increments.
//             // A steps BlockK through m x k
//             // B steps BlockK through k x m
//             auto incrA = MappingA::dataOffset(std::make_pair(0, TILE_DIM), k);
//             auto incrB = MappingB::dataOffset(std::make_pair(TILE_DIM, 0), k);

//             auto endA = addrA + incrA * (k / TILE_DIM);

//             addrA += incrA;
//             addrB += incrB;

//             while(addrA != endA)
//             {
//                 load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
//                 load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);

//                 // Start pulling in the next block
//                 auto fragANext = FragA();
//                 auto fragBNext = FragB();
//                 load_matrix_sync(fragANext, addrA, k);
//                 load_matrix_sync(fragBNext, addrB, k);

//                 // Mma for current block
//                 mma_sync(fragAcc, fragA, fragB, fragAcc);

//                 store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragANext), ldLds);
//                 store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragBNext), ldLds);

//                 addrA += incrA;
//                 addrB += incrB;
//             }

//             // Mma for the last block
//             load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
//             load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);
//             mma_sync(fragAcc, fragA, fragB, fragAcc);

//             // Store fragAcc to global acc
//             auto* accWithOffset = acc + accBatchOffset * blockIdx.z;
//             auto* addrAcc       = MappingAcc::dataCoord(accWithOffset, matrixCoordC, m);
//             store_matrix_sync(addrAcc, fragAcc, m, mem_row_major);

//             // Copy lower triangular from acc to output
//             auto fragColIdx   = threadIdx.x % TILE_DIM;
//             auto globalColIdx = std::get<1>(matrixCoordC) + fragColIdx;
//             auto rowsPerStep  = AMDGCN_WAVE_SIZE / TILE_DIM;

//             count = (TILE_DIM * TILE_DIM) >> Log2<AMDGCN_WAVE_SIZE>::value;
//             for(int i = 0; i < count; i++)
//             {
//                 auto fragRowIdx
//                     = i * rowsPerStep + ((threadIdx.x & (AMDGCN_WAVE_SIZE - 1)) / TILE_DIM);
//                 auto globalRowIdx = std::get<0>(matrixCoordC) + fragRowIdx;
//                 if(globalRowIdx > globalColIdx)
//                 {
//                     auto outputOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);
//                     output[outputBatchOffset * blockIdx.z + outputOffset + globalColIdx]
//                         = static_cast<DataT>(
//                             acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx]);
//                 }
//             }
//         }
//     }

//     template <uint32_t BlockM,
//           uint32_t BlockN,
//           uint32_t BlockK,
//           typename InputT,
//           typename OutputT,
//           typename ComputeT,
//           typename LayoutA,
//           typename LayoutB,
//           typename LayoutC,
//           typename LayoutD>
//     __global__ void __launch_bounds__(256, 1) mmaSyncLds(uint32_t       m,
//                                                         uint32_t       n,
//                                                         uint32_t       k,
//                                                         InputT const*  a,
//                                                         InputT const*  b,
//                                                         OutputT const* c,
//                                                         OutputT*       d,
//                                                         uint32_t       lda,
//                                                         uint32_t       ldb,
//                                                         uint32_t       ldc,
//                                                         uint32_t       ldd,
//                                                         ComputeT       alpha,
//                                                         ComputeT       beta)
//     {
//         // Setup global mapping
//         using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
//         using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
//         using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
//         using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;

//         using FragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
//         using FragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
//         using FragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
//         using FragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

//         // Will store to LDS as though it were a register file.
//         // Rows = register count
//         // Cols = unpacked register elements = 64
//         // Row major to minimize bank conflicts
//         constexpr uint32_t registerFileWidth = AMDGCN_WAVE_SIZE;
//         using MappingLdsA = MappingUtil<FragA::size(), registerFileWidth, InputT, row_major>;
//         using MappingLdsB = MappingUtil<FragB::size(), registerFileWidth, InputT, row_major>;
//         using FragLdsA
//             = wmma::fragment<register_file, 1, registerFileWidth, FragA::size(), InputT, row_major>;
//         using FragLdsB
//             = wmma::fragment<register_file, 1, registerFileWidth, FragB::size(), InputT, row_major>;

//         static_assert(FragA::size() * 64 == BlockM * BlockK, "Elements of A don't match");
//         static_assert(FragLdsA::size() == FragA::size(), "A Sizes don't match");
//         static_assert(FragB::size() * 64 == BlockK * BlockN, "Elements don't match");
//         static_assert(FragLdsB::size() == FragB::size(), "Sizes don't match");

//         // Target C / D block on 2D grid
//         auto matrixCoordC = MappingC::matrixCoord();

//         if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < n && BlockK < k)
//         {
//             // Initialize accumulator
//             auto fragAcc = FragAcc();
//             wmma::fill_fragment(fragAcc, static_cast<ComputeT>(0));

//             // Accumulate A * B
//             if(alpha)
//             {
//                 // Setup starting addresses
//                 auto* addrA = MappingA::dataCoord(a, lda, std::make_pair(std::get<0>(matrixCoordC), 0));
//                 auto* addrB = MappingB::dataCoord(b, ldb, std::make_pair(0, std::get<1>(matrixCoordC)));

//                 // Prefetch the first block from global memory
//                 auto fragA = FragA();
//                 auto fragB = FragB();
//                 wmma::load_matrix_sync(fragA, addrA, lda);
//                 wmma::load_matrix_sync(fragB, addrB, ldb);

//                 // Setup a register file in LDS which is friendly to minimizing bank conflicts.
//                 // Register file blocks in LDS follow same wg mapping for convenience.
//                 // Each wave will prefetch one block of A and one block of B.
//                 // A blocks occupy first portion of LDS and B blocks occupy the latter.
//                 HIP_DYNAMIC_SHARED(void*, localMemPtr);
//                 auto workgroupDim = MappingLdsA::workgroupDim();
//                 auto ldLds        = registerFileWidth * std::get<1>(workgroupDim);

//                 auto* baseAddrLdsA = reinterpret_cast<InputT*>(localMemPtr);
//                 auto* baseAddrLdsB
//                     = baseAddrLdsA + std::get<0>(workgroupDim) * FragLdsA::size() * ldLds;

//                 auto matrixCoordLdsA = MappingLdsA::matrixCoord(MappingLdsA::waveCoord());
//                 auto matrixCoordLdsB = MappingLdsB::matrixCoord(MappingLdsB::waveCoord());

//                 auto* addrLdsA = MappingLdsA::dataCoord(baseAddrLdsA, ldLds, matrixCoordLdsA);
//                 auto* addrLdsB = MappingLdsA::dataCoord(baseAddrLdsB, ldLds, matrixCoordLdsB);

//                 wmma::store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragA), ldLds);
//                 wmma::store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragB), ldLds);

//                 // Setup address increments.
//                 // A steps BlockK through m x k
//                 // B steps BlockK through k x n
//                 auto incrA = MappingA::dataOffset(lda, std::make_pair(0, BlockK));
//                 auto incrB = MappingB::dataOffset(ldb, std::make_pair(BlockK, 0));

//                 auto endA = addrA + incrA * (k / BlockK);

//                 addrA += incrA;
//                 addrB += incrB;

//                 while(addrA != endA)
//                 {
//                     // Keeping the workgroup in sync here is not necessary for correctness.
//                     // HOWEVER, if we keep waves in sync chances are good we may
//                     // benefit from cache hits on re-used data from A and B global loads.
//                     wmma::synchronize_workgroup();
//                     wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
//                     wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);

//                     // Start pulling in the next block
//                     auto fragANext = FragA();
//                     auto fragBNext = FragB();
//                     wmma::load_matrix_sync(fragANext, addrA, lda);
//                     wmma::load_matrix_sync(fragBNext, addrB, ldb);

//                     // Mma for current block
//                     wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

//                     wmma::store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragANext), ldLds);
//                     wmma::store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragBNext), ldLds);

//                     addrA += incrA;
//                     addrB += incrB;
//                 }

//                 // Mma for the last block
//                 wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
//                 wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);
//                 wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
//             }

//             // Load C
//             auto fragC = FragC();
//             wmma::fill_fragment(fragC, static_cast<OutputT>(0));
//             if(beta)
//             {
//                 // Setup address
//                 auto* addrC = MappingC::dataCoord(c, ldc, matrixCoordC);
//                 wmma::load_matrix_sync(fragC,
//                                     addrC,
//                                     ldc,
//                                     std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
//                                                                             : wmma::mem_col_major);
//             }

//             // D = alpha * accumAB + beta * C
// #pragma unroll
//             for(int i = 0; i < fragC.num_elements; ++i)
//             {
//                 fragC.x[i] = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
//             }

//             // Output addresss
//             auto* addrD = MappingD::dataCoord(d, ldd, matrixCoordC);

//             // Store the output
//             wmma::store_matrix_sync(addrD,
//                                     fragC,
//                                     ldd,
//                                     std::is_same<LayoutD, row_major>::value ? wmma::mem_row_major
//                                                                             : wmma::mem_col_major);
//         }
//     }

// } // namespace rocwmma

#endif // DLRM_DOT_FWD_LDS_H
