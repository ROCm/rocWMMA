#ifndef WMMA_DEVICE_MMA_SYNC_H
#define WMMA_DEVICE_MMA_SYNC_H

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
__global__ void __launch_bounds__(256, 1) mmaSync(uint32_t       m,
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
    using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
    using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;

    using FragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
    using FragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
    using FragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
    using FragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

    // Target C / D block on 2D grid
    auto matrixCoordC = MappingC::matrixCoord();

    if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < n)
    {
        // Initialize accumulator
        auto fragAcc = FragAcc();
        wmma::fill_fragment(fragAcc, static_cast<ComputeT>(0));

        // Accumulate A * B
        if(alpha)
        {
            // Setup starting addresses
            auto* addrA = MappingA::dataCoord(a, lda, std::make_pair(std::get<0>(matrixCoordC), 0));
            auto* addrB = MappingB::dataCoord(b, ldb, std::make_pair(0, std::get<1>(matrixCoordC)));

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x n
            auto incrA = MappingA::dataOffset(lda, std::make_pair(0, BlockK));
            auto incrB = MappingB::dataOffset(ldb, std::make_pair(BlockK, 0));

            auto count = k / BlockK;
            for(int i = 0; i < count; i++)
            {
                // Keeping the workgroup in sync here is not necessary for correctness.
                // HOWEVER, if we keep waves in sync chances are good we may
                // benefit from cache hits on re-used data from A and B global loads.
                wmma::synchronize_workgroup();

                auto fragA = FragA();
                auto fragB = FragB();

                // Load and multiply
                wmma::load_matrix_sync(fragA, addrA, lda);
                wmma::load_matrix_sync(fragB, addrB, ldb);
                wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

                addrA += incrA;
                addrB += incrB;
            }
        }

        // Load C
        auto fragC = FragC();
        wmma::fill_fragment(fragC, static_cast<OutputT>(0));
        if(beta)
        {
            // Setup address
            auto* addrC = MappingC::dataCoord(c, ldc, matrixCoordC);
            wmma::load_matrix_sync(fragC,
                                   addrC,
                                   ldc,
                                   std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                           : wmma::mem_col_major);
        }

        // D = alpha * accumAB + beta * C
#pragma unroll
        for(int i = 0; i < fragC.num_elements; ++i)
        {
            fragC.x[i] = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
        }

        // Output addresss
        auto* addrD = MappingD::dataCoord(d, ldd, matrixCoordC);

        // Store the output
        wmma::store_matrix_sync(addrD,
                                fragC,
                                ldd,
                                std::is_same<LayoutD, row_major>::value ? wmma::mem_row_major
                                                                        : wmma::mem_col_major);
    }
}

#endif // WMMA_DEVICE_MMA_SYNC_H
