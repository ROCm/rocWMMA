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

#ifndef WMMA_DEVICE_MMA_SYNC_MULTI_LDS_H
#define WMMA_DEVICE_MMA_SYNC_MULTI_LDS_H

#include <algorithm>
#include <array>
#include <utility>

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "LdsMappingUtil.h"
#include "WMMA.h"
#include "WMMACoop.h"
#pragma GCC diagnostic pop

namespace rocwmma
{

    // A few workarounds for some lack of std library support on GPU
    template <typename T>
    __device__ inline void assign(T& m, T const& val)
    {
        m = val;
    }

    template <typename T1, typename T2>
    __device__ inline void assign(std::pair<T1, T2>& m, std::pair<T1, T2> const& val)
    {
        std::get<0>(m) = std::get<0>(val);
        std::get<1>(m) = std::get<1>(val);
    }

    template <typename T, size_t X>
    __device__ inline void assign(std::array<T, X>& m, int32_t x, T const& val)
    {
        assign(const_cast<T&>(m[x]), val);
    }

    template <typename T, size_t X, size_t Y>
    __device__ inline void
        assign(std::array<std::array<T, Y>, X>& m, int32_t x, int32_t y, T const& val)
    {
        assign(const_cast<std::array<T, Y>&>(m[x]), y, val);
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD,
              typename LayoutLds,
              typename LdsMapping,
              uint32_t BlocksX = 1,
              uint32_t BlocksY = 1>
    __global__ void __launch_bounds__(256, 1) mmaSyncMultiLds(uint32_t       m,
                                                              uint32_t       n,
                                                              uint32_t       k,
                                                              InputT const*  a,
                                                              InputT const*  b,
                                                              OutputT const* c,
                                                              OutputT*       d,
                                                              uint32_t       lda,
                                                              uint32_t       ldb,
                                                              uint32_t       ldc,
                                                              uint32_t       ldd,
                                                              ComputeT       alpha,
                                                              ComputeT       beta)
    {
        // Setup global mapping
        using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
        using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
        using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
        using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;
        using CoordT   = typename MappingA::MatrixCoordT;

        using FragA   = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
        using FragB   = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
        using FragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
        using FragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

        // Will store to LDS as though it were a register file.
        // Rows = register count
        // Cols = unpacked register elements = 64
        // Row major to minimize bank conflicts

        using MappingLds = LdsMappingUtil<BlockM,
                                          BlockN,
                                          BlockK,
                                          InputT,
                                          LayoutA,
                                          LayoutB,
                                          LayoutLds,
                                          LdsMapping,
                                          BlocksX,
                                          BlocksY>;

        using GlobalReadFragA = typename MappingLds::GlobalReadFragA;
        using GlobalReadFragB = typename MappingLds::GlobalReadFragB;

        using LocalWriteFragA = typename MappingLds::LocalWriteFragA;
        using LocalWriteFragB = typename MappingLds::LocalWriteFragB;

        using LocalReadFragA = typename MappingLds::LocalReadFragA;
        using LocalReadFragB = typename MappingLds::LocalReadFragB;

        // Target starting C / D block on 2D grid, offset by blocks per wave
        auto matrixCoordC = MappingC::matrixCoord();
        std::get<0>(matrixCoordC) *= BlocksX;
        std::get<1>(matrixCoordC) *= BlocksY;

        if((std::get<0>(matrixCoordC) + BlocksX * BlockM) <= m
           && (std::get<1>(matrixCoordC) + BlocksY * BlockN) <= n && (BlockK < k))
        {
            typename MappingC::MatrixCoordT subMatrixCoordsC[BlocksX][BlocksY];
            FragAcc                         fragsAccum[BlocksX][BlocksY];

            /// Setup LDS addressing and start writing pre-fetch to LDS
            ///

            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + MappingLds::ldsWidth() * MappingLds::ldsHeight();

            ///
            /// Initialize sub matrix coords and accum frags
            ///
#pragma unroll
            for(int i = 0; i < BlocksX; ++i)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; ++j)
                {
                    // Initialize sub matrix coordinates
                    auto subMatCoordC = matrixCoordC;
                    std::get<0>(subMatCoordC) += i * BlockM;
                    std::get<1>(subMatCoordC) += j * BlockN;
                    assign(subMatrixCoordsC[i][j], subMatCoordC);

                    // Initialize accumulators
                    auto fragAcc = FragAcc();
                    fill_fragment(fragAcc, static_cast<ComputeT>(0));
                    fragsAccum[i][j] = fragAcc;
                }
            }

            MappingLds::prefetchGlobalA(
                ldsPtrLo,
                MappingA::dataCoord(a, std::make_pair(std::get<0>(subMatrixCoordsC[0][0]), 0), lda),
                lda);

            MappingLds::prefetchGlobalB(
                ldsPtrLo,
                MappingB::dataCoord(b, std::make_pair(0, std::get<1>(subMatrixCoordsC[0][0])), ldb),
                ldb);

            ///
            /// Accumulate A * B
            ///
            if(alpha)
            {
                // Wait for A / B write LDS
                synchronize_workgroup();

                ///
                /// Step through and accumulate A * B
                ///
                for(int currentK = BlockK; currentK < k; currentK += BlockK)
                {
                    FragA cachedFragsA[BlocksX];
                    FragB cachedFragsB[BlocksY];

                    // Cache lds blocks to register
                    MappingLds::prefetchLocalA(cachedFragsA, ldsPtrLo);
                    MappingLds::prefetchLocalB(cachedFragsB, ldsPtrLo);

                    MappingLds::prefetchGlobalA(
                        ldsPtrHi,
                        MappingA::dataCoord(
                            a, std::make_pair(std::get<0>(subMatrixCoordsC[0][0]), currentK), lda),
                        lda);

                    MappingLds::prefetchGlobalB(
                        ldsPtrHi,
                        MappingB::dataCoord(
                            b, std::make_pair(currentK, std::get<1>(subMatrixCoordsC[0][0])), ldb),
                        ldb);

                    // A * B
#pragma unroll
                    for(int i = 0; i < BlocksX; i++)
                    {
#pragma unroll
                        for(int j = 0; j < BlocksY; j++)
                        {
                            mma_sync(const_cast<FragAcc&>(fragsAccum[i][j]),
                                     cachedFragsA[i],
                                     cachedFragsB[j],
                                     fragsAccum[i][j]);
                        }
                    }

                    // Wait for A / B read LDS
                    synchronize_workgroup();

                    //std::swap(reinterpret_cast<float32_t*>(ldsPtrLo), reinterpret_cast<float32_t*>(ldsPtrHi));
                    auto* tmp = ldsPtrLo;
                    ldsPtrLo  = ldsPtrHi;
                    ldsPtrHi  = tmp;
                }

                ///
                /// Clean up tail MMA
                ///

                // Tail A * B
                FragA cachedFragsA[BlocksX];
                FragB cachedFragsB[BlocksY];

                // Cache lds blocks to register
                MappingLds::prefetchLocalA(cachedFragsA, ldsPtrLo);
                MappingLds::prefetchLocalB(cachedFragsB, ldsPtrLo);

                // A * B
#pragma unroll
                for(int i = 0; i < BlocksX; i++)
                {
#pragma unroll
                    for(int j = 0; j < BlocksY; j++)
                    {
                        mma_sync(
                            fragsAccum[i][j], cachedFragsA[i], cachedFragsB[j], fragsAccum[i][j]);
                    }
                }
            }

            ///
            /// Initialize C frags
            ///

            FragC fragsC[BlocksX][BlocksY];

#pragma unroll
            for(int i = 0; i < BlocksX; ++i)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; ++j)
                {
                    // Initialize C frags
                    auto fragC = FragC();
                    fill_fragment(fragC, static_cast<OutputT>(0));
                    fragsC[i][j] = fragC;
                }
            }

            if(beta)
            {
#pragma unroll
                for(int i = 0; i < BlocksX; i++)
                {
#pragma unroll
                    for(int j = 0; j < BlocksY; j++)
                    {
                        auto* addrC = MappingC::dataCoord(c, subMatrixCoordsC[i][j], ldc);
                        load_matrix_sync(fragsC[i][j],
                                         addrC,
                                         ldc,
                                         std::is_same<LayoutC, row_major>::value ? mem_row_major
                                                                                 : mem_col_major);
                    }
                }
            }

// D = alpha * accumAB + beta * C
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    auto& fragAcc = fragsAccum[i][j];
                    auto& fragC   = fragsC[i][j];

#pragma unroll
                    for(int k = 0; k < fragC.num_elements; ++k)
                    {
                        fragC.x[k]
                            = OutputT(alpha * ComputeT(fragAcc.x[k]) + beta * ComputeT(fragC.x[k]));
                    }

                    // Output addresss
                    auto* addrD = MappingD::dataCoord(d, subMatrixCoordsC[i][j], ldd);

                    // Store the output
                    store_matrix_sync(addrD,
                                      fragC,
                                      ldd,
                                      std::is_same<LayoutD, row_major>::value ? mem_row_major
                                                                              : mem_col_major);
                }
            }
        }
    }

} // namespace rocwmma

#endif // WMMA_DEVICE_MMA_SYNC_MULTI_LDS_H
