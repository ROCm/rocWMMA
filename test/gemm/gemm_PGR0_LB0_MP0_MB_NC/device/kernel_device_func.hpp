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

#ifndef ROCWMMA_GEMM_TEST_DEVICE_FUNC
#define ROCWMMA_GEMM_TEST_DEVICE_FUNC

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    ///
    /// This class of kernel is an extension of the naive kernel whereas
    /// each wave is responsible for calculating a macro tile area of
    /// BlocksX * BlockM x BlocksY * BlockN
    ///
    /// Kernel behaviour is described by:
    /// PGR0 = Prefetch Global Read = 0, no prefetch
    /// LB0 = Lds Blocks = 0, no Lds usage
    /// MP0 = Mfma Priority = 0, no setprio
    /// MB = Multi-block, BlocksX * BlocksY > 1
    /// NC = Non-cooperative
    ///
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
    __global__ void __launch_bounds__(256) gemm_PGR0_LB0_MP0_MB_NC(uint32_t       m,
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
        using FragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutC>;
        using FragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutD>;

        // Target starting C / D block on 2D grid, offset by blocks per wave
        auto matrixCoordC = MappingC::matrixCoord();
        matrixCoordC.x *= BlocksX;
        matrixCoordC.y *= BlocksY;

        if(matrixCoordC.x + BlocksX * BlockM > m || matrixCoordC.y + BlocksY * BlockN > n)
        {
            return;
        }

        if(BlockK > k)
        {
            return;
        }

        // Targets for C sub-matrices
        CoordT  subMatrixCoordsC[BlocksX][BlocksY];
        FragAcc fragsAccum[BlocksX][BlocksY];

        /// Initialization
#pragma unroll
        for(int i = 0; i < BlocksX; i++)
        {
#pragma unroll
            for(int j = 0; j < BlocksY; j++)
            {
                // Initialize sub matrix coordinates
                subMatrixCoordsC[i][j].x = matrixCoordC.x + i * BlockM;
                subMatrixCoordsC[i][j].y = matrixCoordC.y + j * BlockN;

                // Initialize accumulators
                fill_fragment(fragsAccum[i][j], static_cast<ComputeT>(0));
            }
        }

        /// Setup global read addresses
        InputT const* globalAddrsA[BlocksX];
        InputT const* globalAddrsB[BlocksY];

        // Blocks in the same row share the same starting address for A
#pragma unroll
        for(int i = 0; i < BlocksX; i++)
        {
            globalAddrsA[i] = MappingA::dataCoord(a, Coord2d(subMatrixCoordsC[i][0].x, 0), lda);
        }

        // Blocks in the same col share the same starting address for B
#pragma unroll
        for(int i = 0; i < BlocksY; i++)
        {
            globalAddrsB[i] = MappingB::dataCoord(b, Coord2d(0, subMatrixCoordsC[0][i].y), ldb);
        }

        /// Setup address increments.
        // A steps BlockK through m x k
        // B steps BlockK through k x n
        auto incrA  = MappingA::dataOffset(Coord2d(0, BlockK), lda);
        auto incrB  = MappingB::dataOffset(Coord2d(BlockK, 0), ldb);
        auto stepsK = k / BlockK;

        /// Accumulate A * B
        for(int currentStep = 0; currentStep < stepsK; currentStep++)
        {
            // Cache the B-Blocks for re-use
            FragB cachedFragsB[BlocksY];
            FragA fragA;

            // Synchronize workgroup increases chances for cache hits
            synchronize_workgroup();

#pragma unroll
            for(int j = 0; j < BlocksY; j++)
            {
                load_matrix_sync(cachedFragsB[j], globalAddrsB[j], ldb);
                globalAddrsB[j] += incrB;
            }

            //#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                // A fragment will be re-used for each B
                load_matrix_sync(fragA, globalAddrsA[i], lda);
                globalAddrsA[i] += incrA;

                //#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    mma_sync(fragsAccum[i][j], fragA, cachedFragsB[j], fragsAccum[i][j]);
                }
            }
        }

        // Initialize C frags
        FragC fragsC[BlocksX][BlocksY];

#pragma unroll
        for(int i = 0; i < BlocksX; i++)
        {
#pragma unroll
            for(int j = 0; j < BlocksY; j++)
            {
                auto* addrC = MappingC::dataCoord(c, subMatrixCoordsC[i][j], ldc);
                load_matrix_sync(fragsC[i][j], addrC, ldc);
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
                for(int e = 0; e < fragC.num_elements; ++e)
                {
                    fragC.x[e]
                        = OutputT(alpha * ComputeT(fragAcc.x[e]) + beta * ComputeT(fragC.x[e]));
                }

                // Output addresss
                auto* addrD = MappingD::dataCoord(d, subMatrixCoordsC[i][j], ldd);

                // Store the output
                store_matrix_sync(addrD, fragC, ldd);
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_FUNC
