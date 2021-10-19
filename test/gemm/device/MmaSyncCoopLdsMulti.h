/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#ifndef WMMA_DEVICE_MMA_SYNC_COOP_LDSM_MULTI_H
#define WMMA_DEVICE_MMA_SYNC_COOP_LDSM_MULTI_H

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "WMMA.h"
#pragma GCC diagnostic pop

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
          uint32_t BlocksX = 1,
          uint32_t BlocksY = 1>
__global__ void __launch_bounds__(256, 1) mmaSyncTestCoopLdsMulti(uint32_t       m,
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
    using CoordT   = typename MappingA::CoordT;

    using FragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
    using FragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
    using FragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
    using FragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

    // Will store to LDS as though it were a register file.
    // Rows = register count
    // Cols = unpacked register elements = 64
    // Row major to minimize bank conflicts
    constexpr uint32_t registerFileWidth = AMDGCN_WAVE_SIZE;
    using MappingLdsA = MappingUtil<FragA::size(), registerFileWidth, InputT, row_major>;
    using MappingLdsB = MappingUtil<FragB::size(), registerFileWidth, InputT, row_major>;
    using FragLdsA    = wmma::
        fragment<register_file_coop_a, 1, registerFileWidth, FragA::size(), InputT, row_major>;
    using FragLdsB = wmma::
        fragment<register_file_coop_b, 1, registerFileWidth, FragB::size(), InputT, row_major>;

    static_assert(FragA::size() * registerFileWidth == BlockM * BlockK,
                  "Elements of A don't match");
    static_assert(FragLdsA::size() == FragA::size(), "A Sizes don't match");
    static_assert(FragB::size() * registerFileWidth == BlockK * BlockN, "Elements don't match");
    static_assert(FragLdsB::size() == FragB::size(), "Sizes don't match");

    // Target C / D block on 2D grid, offset by blocks per wave
    auto matrixCoordC = MappingC::matrixCoord();
    std::get<0>(matrixCoordC) *= BlocksX;
    std::get<1>(matrixCoordC) *= BlocksY;

    if((std::get<0>(matrixCoordC) * BlocksX) < m && (std::get<1>(matrixCoordC) * BlocksY) < n
       && BlockK < k)
    {
        // Targets for C sub-matrices
        std::array<std::array<typename MappingC::CoordT, BlocksY>, BlocksX> subMatrixCoordsC;
        std::array<std::array<FragAcc, BlocksY>, BlocksX>                   fragsAccum;
        std::array<std::array<FragC, BlocksY>, BlocksX>                     fragsC;

#pragma unroll
        for(int i = 0; i < BlocksX; i++)
        {
#pragma unroll
            for(int j = 0; j < BlocksY; j++)
            {
                // Initialize sub matrix coordinates
                auto subMatCoordC = matrixCoordC;
                std::get<0>(subMatCoordC) += i * BlockM;
                std::get<1>(subMatCoordC) += j * BlockN;
                assign(subMatrixCoordsC, i, j, subMatCoordC);

                // Initialize accumulators
                auto fragAcc = FragAcc();
                wmma::fill_fragment(fragAcc, static_cast<ComputeT>(0));
                assign(fragsAccum, i, j, fragAcc);

                // Initialize C frags
                auto fragC = FragC();
                wmma::fill_fragment(fragC, static_cast<OutputT>(0));
                assign(fragsC, i, j, fragC);
            }
        }

        // Accumulate A * B
        if(alpha)
        {
            // Setup global read addresses
            std::array<InputT*, BlocksX> globalAddrsA;
            std::array<InputT*, BlocksY> globalAddrsB;

// Blocks in the same row share the same starting address for A
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                auto subMatARowRef = subMatrixCoordsC[i][0];
                assign(globalAddrsA,
                       i,
                       MappingA::dataCoord(a, lda, std::make_pair(std::get<0>(subMatARowRef), 0)));
            }

// Blocks in the same col share the same starting address for B
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                auto subMatBColRef = subMatrixCoordsC[0][i];
                assign(globalAddrsB,
                       i,
                       MappingB::dataCoord(b, ldb, std::make_pair(0, std::get<1>(subMatBColRef))));
            }

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x n
            auto incrA        = MappingA::dataOffset(lda, std::make_pair(0, BlockK));
            auto incrB        = MappingB::dataOffset(ldb, std::make_pair(BlockK, 0));
            auto stepsK       = k / BlockK;
            auto currentSteps = 0;

            for(int currentStep = 0; currentStep < stepsK; currentStep++)
            {
                // Cache the B-Blocks for re-use
                std::array<FragB, BlocksY> cachedFragsB;

#pragma unroll
                for(int i = 0; i < BlocksX; i++)
                {
                    wmma::synchronize_workgroup();

                    // A fragment will be re-used for each B
                    auto fragA = FragA();
                    wmma::load_matrix_sync(fragA, globalAddrsA[i], lda);

#pragma unroll
                    for(int j = 0; j < BlocksY; j++)
                    {
                        // Need to load the B fragments only once.
                        if(i == 0)
                        {
                            wmma::load_matrix_sync(
                                const_cast<FragB&>(cachedFragsB[j]), globalAddrsB[j], ldb);
                            const_cast<InputT*&>(globalAddrsB[j]) += incrB;
                        }

                        wmma::mma_sync(const_cast<FragAcc&>(fragsAccum[i][j]),
                                       fragA,
                                       cachedFragsB[j],
                                       fragsAccum[i][j]);
                    }

                    const_cast<InputT*&>(globalAddrsA[i]) += incrA;
                }
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
                    auto* addrC = MappingC::dataCoord(c, ldc, subMatrixCoordsC[i][j]);
                    wmma::load_matrix_sync(const_cast<FragC&>(fragsC[i][j]),
                                           addrC,
                                           ldc,
                                           std::is_same<LayoutC, row_major>::value
                                               ? wmma::mem_row_major
                                               : wmma::mem_col_major);
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
                auto& fragAcc = const_cast<FragAcc&>(fragsAccum[i][j]);
                auto& fragC   = const_cast<FragC&>(fragsC[i][j]);

#pragma unroll
                for(int i = 0; i < fragC.num_elements; ++i)
                {
                    fragC.x[i]
                        = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
                }

                // Output addresss
                auto* addrD = MappingD::dataCoord(d, ldd, subMatrixCoordsC[i][j]);

                // Store the output
                wmma::store_matrix_sync(addrD,
                                        fragC,
                                        ldd,
                                        std::is_same<LayoutD, row_major>::value
                                            ? wmma::mem_row_major
                                            : wmma::mem_col_major);
            }
        }
    }
}

#endif // WMMA_DEVICE_MMA_SYNC_COOP_LDSM_MULTI_H
