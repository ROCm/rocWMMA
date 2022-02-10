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

#ifndef WMMA_DEVICE_BARRIER_H
#define WMMA_DEVICE_BARRIER_H

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <WMMA/WMMA.h>
#pragma GCC diagnostic pop

namespace rocwmma
{

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    __global__ void __launch_bounds__(256, 1) barrierTest(uint32_t       m,
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

        using FragA   = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
        using FragB   = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
        using FragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
        using FragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

        // Target C / D block on 2D grid
        auto matrixCoordC = MappingC::matrixCoord();

        if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < n)
        {
            // Initialize accumulator
            auto fragAcc = FragAcc();
            fill_fragment(fragAcc, static_cast<ComputeT>(0));

            // Accumulate A * B
            if(alpha)
            {
                // Setup starting addresses
                auto* addrA = MappingA::dataCoord(a, MappingC::matrixCoordN(0), lda);
                auto* addrB = MappingB::dataCoord(b, MappingC::matrixCoordM(0), ldb);

                // Setup address increments.
                // A steps BlockK through m x k
                // B steps BlockK through k x n
                auto incrA = MappingA::dataOffset(std::make_pair(0, BlockK), lda);
                auto incrB = MappingB::dataOffset(std::make_pair(BlockK, 0), ldb);

                auto count = k / BlockK;
                for(int i = 0; i < count; i++)
                {
                    // Keeping the workgroup in sync here is not necessary for correctness.
                    // HOWEVER, if we keep waves in sync chances are good we may
                    // benefit from cache hits on re-used data from A and B global loads.
                    synchronize_workgroup();

                    auto fragA = FragA();
                    auto fragB = FragB();

                    // Load and multiply
                    load_matrix_sync(fragA, addrA, lda);
                    load_matrix_sync(fragB, addrB, ldb);
                    mma_sync(fragAcc, fragA, fragB, fragAcc);

                    addrA += incrA;
                    addrB += incrB;
                }
            }

            // Load C
            auto fragC = FragC();
            fill_fragment(fragC, static_cast<OutputT>(0));
            if(beta)
            {
                // Setup address
                auto* addrC = MappingC::dataCoord(c, matrixCoordC, ldc);
                load_matrix_sync(fragC,
                                 addrC,
                                 ldc,
                                 std::is_same<LayoutC, row_major>::value ? mem_row_major
                                                                         : mem_col_major);
            }

            // D = alpha * accumAB + beta * C
#pragma unroll
            for(int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
            }

            // Output addresss
            auto* addrD = MappingD::dataCoord(d, matrixCoordC, ldd);

            // Store the output
            store_matrix_sync(addrD,
                              fragC,
                              ldd,
                              std::is_same<LayoutD, row_major>::value ? mem_row_major
                                                                      : mem_col_major);
        }
    }

} // namespace rocwmma

#endif // WMMA_DEVICE_BARRIER_H
