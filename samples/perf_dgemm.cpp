/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma_coop.hpp>

#include "common.hpp"

using rocwmma::float64_t;

//Best permorming Block Sizes
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;

// Supports ROCWMMA_K sizes as
// : multiples of 16.
const int ROCWMMA_K = 16;

// Device warp size
const uint32_t WAVE_SIZE = getWarpSize();

// Per-wave output block coverage
const int BLOCKS_X = 2;
const int BLOCKS_Y = 2;

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute BLOCKS_X * ROCWMMA_M x BLOCKS_Y * ROCWMMA_N
// output elements
// Note: Workgroup will compute
// T_BLOCK_X * BLOCKS_X * BLOCK_M / WAVE_SIZE x T_BLOCK_Y * BLOCKS_Y * ROCWMMA_N
// output elements
const int T_BLOCK_X = 128;
const int T_BLOCK_Y = 2;

using namespace rocwmma;

// Matrix data initialization
template <typename DataT, typename Layout>
__host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto index = std::is_same<Layout, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto ld    = std::is_same<Layout, rocwmma::row_major>::value ? n : m;

#pragma omp parallel for
    for(int i = 0; i < m; ++i) // row
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j) // col
        {
            // Count up in integers, in ascending order for each row.
            auto value = (i * n + j) % 5;
            auto idx   = index(i, j, ld);
            mat[idx]   = ((value % 3) && std::is_signed<DataT>::value)
                                ? -static_cast<DataT>(value)
                                : static_cast<DataT>(value);
        }
    }
}

// Host GEMM validation
template <typename InputT, typename OutputT, typename ComputeT>
__host__ void gemm_cpu_host(uint32_t       m,
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
#pragma omp parallel for
    for(int i = 0; i < m; ++i) // row
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j) // col
        {
            ComputeT accum = 0.0f;
            auto idx = j * ldc + i;
            for(int h = 0; h < k; ++h)
            {
                auto idxA = h * lda + i;
                auto idxB = h * ldb + j;
                accum += static_cast<ComputeT>(a[idxA])
                         * static_cast<ComputeT>(b[idxB]);
            }
             d[idx] = alpha * accum + beta * c[idx];
        }
    }
}

// Cooperative Load Matrix - Loads the entire fragment with data from memory address cooperatively across waves.
// Each cooperative wave is responsible in loading a portion of the final fragment.
// Note that the full fragment data is not cohesive for individual waves as they only load a piece of the data.
// This function may be paired with store_matrix_sync to move a single fragment collaboratively between memory locations.
//
// The full load is split into work items (splitCount).
// Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
// The current wave index determines the order of the current wave in the collaboration pool.
// Work items are consumed in order by waves [0, waveCount) until
// there are no more work items and the operation is completed.

// Global A reads in cooperative mode
template <uint32_t waveCountA, uint32_t splitCountA, typename GRFragA>
__device__ static inline void globalReadCoopA(GRFragA& grFragA, 
                                              float64_t const* gAddrA,
                                              uint32_t lda, 
                                              uint32_t waveIndexA)
{
    rocwmma::template load_matrix_coop_sync <waveCountA, splitCountA>(
                                             grFragA, gAddrA, lda, waveIndexA);
}

// Global B reads in cooperative mode
template <uint32_t waveCountB, uint32_t splitCountB, typename GRFragB>
__device__ static inline void globalReadCoopB(GRFragB &grFragB,
                                              float64_t const* gAddrB,
                                              uint32_t ldb, 
                                              uint32_t waveIndexB)
{
    rocwmma::template load_matrix_coop_sync <waveCountB, splitCountB>(
                                             grFragB, gAddrB, ldb, waveIndexB);
}

// Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
// Each cooperative wave is responsible in storing a portion of the final fragment.
// Note that the full fragment data is not required to be cohesive for individual waves as they
// only store a piece of the data. This function may be paired with load_matrix_sync to move a single fragment
// collaboratively between memory locations.
//
// The full store is split into work items (splitCount). Work items are assigned
// in round robin fashion to waves in the range of [0, waveCount). The current
// wave index determines the order of the current wave in the collaboration pool.
// Work items are consumed in order by waves [0, waveCount) until there are no more
// work items and the operation is completed.

// Local A writes in cooperative mode
template <uint32_t waveCountA, uint32_t splitCountA, typename LWFragA>
__device__ static inline void localWriteCoopA(float64_t* ldsAddr,
                                              LWFragA const &lwFragA,
                                              uint32_t ldsWidth,
                                              uint32_t waveIndexA)
{
     rocwmma::template store_matrix_coop_sync <waveCountA, splitCountA>(
                                               ldsAddr, lwFragA, ldsWidth, waveIndexA);
}

// Local B writes in cooperative mode
template <uint32_t waveCountB, uint32_t splitCountB, typename LWFragB>
__device__ static inline void localWriteCoopB(float64_t* ldsAddr,
                                              LWFragB const& lwFragB,
                                              uint32_t ldsWidth,
                                              uint32_t waveIndexB)
{
    rocwmma::template store_matrix_coop_sync <waveCountB, splitCountB>(
                                              ldsAddr, lwFragB, ldsWidth, waveIndexB);
}

// Local A read non-cooperative
// BLOCKS_X frags
template <typename MfmaFragA>
__device__ static inline void localReadA(MfmaFragA (&fragsA)[BLOCKS_X],
                                         float64_t const* ldsAddrA,
                                         uint32_t LdsWidth)
{
    auto readOffset = make_coord2d(ROCWMMA_M, 0u);
    auto blockStep = get<0>(readOffset) * LdsWidth + get<1>(readOffset);

    using LRFragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, row_major>;
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        rocwmma::load_matrix_sync(reinterpret_cast<LRFragA&>(fragsA[i]),
                                  ldsAddrA,
                                  LdsWidth);

        ldsAddrA += blockStep;
    }
}

// Local B read non-cooperative
// BLOCKS_Y frags
template <typename MfmaFragB>
__device__ static inline void localReadB(MfmaFragB (&fragsB)[BLOCKS_Y],
                                         float64_t const* ldsAddrB,
                                         uint32_t LdsWidth)
{
    auto readOffset = make_coord2d(0u, ROCWMMA_N);
    auto blockStep = get<0>(readOffset) * LdsWidth + get<1>(readOffset);

    using LRFragB = fragment<matrix_a, ROCWMMA_N, ROCWMMA_M, ROCWMMA_K, float64_t, row_major>;
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        rocwmma::load_matrix_sync(reinterpret_cast<LRFragB&>(fragsB[i]),
                                  ldsAddrB,
                                  LdsWidth);
        ldsAddrB += blockStep;
    }
}

// Global C reads non-cooperative
// BLOCKS_X * BLOCKS_Y fragments
template <typename MfmaFragC>
__device__ static inline void globalReadC(MfmaFragC (&fragC)[BLOCKS_X][BLOCKS_Y],
                                          float64_t const* gAddrC,
                                          uint32_t ldc)
{
    auto readOffset = make_coord2d(ROCWMMA_M, 0u);
    auto blockStepX = get<1>(readOffset) * ldc + get<0>(readOffset);

    readOffset = make_coord2d(0u, ROCWMMA_N);
    auto blockStepY = get<1>(readOffset) * ldc + get<0>(readOffset);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            rocwmma::load_matrix_sync(fragC[i][j], gAddrC + offsetY, ldc);
            offsetY += blockStepY;
        }
        gAddrC += blockStepX;
    }
}

// Global D writes non-cooperative
// BLOCKS_X * BLOCKS_Y fragments
template <typename MfmaFragD>
__device__ static inline void globalWriteD(float64_t* gAddrD,
                                           MfmaFragD const (&fragsD)[BLOCKS_X][BLOCKS_Y],
                                           uint32_t ldd)
{
    auto writeOffset = make_coord2d(ROCWMMA_M, 0u);
    auto blockStepX = get<1>(writeOffset) * ldd + get<0>(writeOffset);

    writeOffset = make_coord2d(0u, ROCWMMA_N);
    auto blockStepY = get<1>(writeOffset) * ldd + get<0>(writeOffset);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            rocwmma::store_matrix_sync(gAddrD + offsetY, fragsD[i][j], ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

// Broadcast value to fragment
// BLOCKS_X * BLOCKS_Y frags
template <typename FragT>
__device__ static inline void fill(FragT (&frags)[BLOCKS_X][BLOCKS_Y],
                                   float64_t value)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            fill_fragment(frags[i][j], value);
        }
    }
}

// Performs mfma
// BLOCKS_X * BLOCKS_Y frags
__device__  void mfma(
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t> (&fragAccOut)[BLOCKS_X][BLOCKS_Y],
    fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> const (&fragA)[BLOCKS_X],
    fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, row_major> const (&fragB)[BLOCKS_Y],
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t> const (&fragAccIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            rocwmma::mma_sync(fragAccOut[i][j], fragA[i], fragB[j], fragAccIn[i][j]);
        }
    }
}

/// Uniform fused multiply - add (FMA)
// Performs D = alpha * acc + beta * C, where alpha, beta are uniform scalars
__device__  void uniformFma(
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> (&fragsD)[BLOCKS_X][BLOCKS_Y],
    float32_t alpha,
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t> const (&fragsAcc)[BLOCKS_X][BLOCKS_Y],
    float32_t beta,
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> const (&fragsC)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            for(int k = 0; k < fragsD[i][j].num_elements; k++)
            {
                // Perform computation in ComputeT and cast back to OutputT
                fragsD[i][j].x[k] = static_cast<float64_t>(
                                    alpha * fragsAcc[i][j].x[k] +
                                    beta * fragsC[i][j].x[k]);
            }
        }
    }
}


// The following GEMM device kernel implementation is a multi-block GEMM 
// where each wave is responsible for a BLOCKS_X x BLOCKS_Y grid of output blocks.
// This kernel leverages shared memory to implement a data prefetching pipeline
// and collaborates with other waves to improve performance.
// Implements single stage prefetch, double lds buffers, default MFMA prioritization,
// multiple blocks output and is macro-tile collaborative in global read / local write.
// Each wave will compute BLOCKS_X * ROCWMMA_M x BLOCKS_Y * ROCWMMA_N 
// output elements of the M x N x K GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this simplified example, we assume:
// : A is in col-major format     (K x M)
// : B is in row-major format     (N x K)
// : C, D are in col-major format (N x M)
// : Multiplication is NOT in-place, output is written to D matrix
// : used LDS
__global__ void __launch_bounds__(256) gemm_rocwmma_d(uint32_t         m,
                                                      uint32_t         n,
                                                      uint32_t         k,
                                                      float64_t const* a,
                                                      float64_t const* b,
                                                      float64_t const* c,
                                                      float64_t*       d,
                                                      uint32_t         lda,
                                                      uint32_t         ldb,
                                                      uint32_t         ldc,
                                                      uint32_t         ldd,
                                                      float32_t        alpha,
                                                      float32_t        beta)
{
    // Global workgroup Id
    uint32_t wgCoord_x = blockIdx.x;
    uint32_t wgCoord_y = blockIdx.y;

    // Local wave coordinate relative to current workgroup.
    uint32_t localWaveCoord_x = threadIdx.x / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    uint32_t localWaveCoord_y = threadIdx.y;
    
    // Size of workgroup, normalized to wave count.
    constexpr static uint32_t wgDim_x = T_BLOCK_X / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    constexpr static uint32_t wgDim_y = T_BLOCK_Y;
               
    // Wave tile = tile processed by current wave
    uint32_t waveTileSize_x = ROCWMMA_M * BLOCKS_X;
    uint32_t waveTileSize_y = ROCWMMA_N * BLOCKS_Y;

    // Macro tile = tile processed by entire workgroup
    uint32_t macroTileSizeC_x = wgDim_x * waveTileSize_x;
    uint32_t macroTileSizeC_y = wgDim_y * waveTileSize_y;

    // Global matrix coordinate of macro tile for the current workgroup
    uint32_t macroTileCoordC_x = wgCoord_x * macroTileSizeC_x;
    uint32_t macroTileCoordC_y = wgCoord_y * macroTileSizeC_y;

    // The offset from macro tile to the mfma tiles for current wave
    uint32_t waveOffsetC_x = localWaveCoord_x * waveTileSize_x;
    uint32_t waveOffsetC_y = localWaveCoord_y * waveTileSize_y;

    // Global matrix coordinate of wave tile for the current wave
    uint32_t waveTileCoord_x = macroTileCoordC_x + waveOffsetC_x;
    uint32_t waveTileCoord_y = macroTileCoordC_y + waveOffsetC_y;

    uint32_t waveTileBound_x = waveTileCoord_x + waveTileSize_x;
    uint32_t waveTileBound_y = waveTileCoord_y + waveTileSize_y;

    // Bounds check
    if(waveTileBound_x > m || waveTileBound_y > n)
    {
        return;
    }
    
    // Create frags
    // Fragment A computes T_BLOCK_X / WAVE_SIZE * BLOCKS_X * ROCWMMA_M x ROCWMMA_K output elements
    rocwmma::fragment<rocwmma::matrix_a,
                      T_BLOCK_X / rocwmma::Constants::AMDGCN_WAVE_SIZE * BLOCKS_X * ROCWMMA_M,
                      ROCWMMA_N,
                      ROCWMMA_K,
                      float64_t,
                      rocwmma::col_major> grBuffA;

    // Fragment B computes T_BLOCK_Y * BLOCKS_Y * ROCWMMA_N x ROCWMMA_K output elements
    rocwmma::fragment<rocwmma::matrix_b, ROCWMMA_M, T_BLOCK_Y * BLOCKS_Y * ROCWMMA_N,
                ROCWMMA_K, float64_t, rocwmma::row_major> grBuffB;

    ///
    /// Setup global addressing offsets in 1D
    ///   
    auto globalReadOffsetA = macroTileCoordC_x;
    auto globalReadOffsetB = macroTileCoordC_y;
    auto globalReadOffsetC = (waveTileCoord_y * ldc) + waveTileCoord_x;
    auto globalWriteOffsetD = (waveTileCoord_y * ldd) + waveTileCoord_x;

    auto kStepOffsetA =  ROCWMMA_K * lda;
    auto kStepOffsetB =  ROCWMMA_K * ldb;

    // All waves are collaborative.
    // Scheduling order is analogous to row major priority.
    // E.g. Wg = (128, 2) = 2x2 waves
    // (0, 0)   (0, 1)   Share Schedule: i0 = (0, 0), i1 = (0, 1),
    // (1, 0)   (1, 1)                   i2 = (1, 0), i3 = (1, 1), count = 4

    // waveCount is the Number of waves assigned for collaboration
    constexpr static auto waveCount = wgDim_x * wgDim_y;
    // waveIndex is the Index assignment of current wave in collaboration
    auto waveIndex = (localWaveCoord_x * wgDim_y) + localWaveCoord_y;

    // Create LDS fragments and load from global fragments
    // LdsMappingNT (Block Width = LDS Width = ROCWMMA_K)
    // Matrix geometry for A and B have a common dimension (ROCWMMA_K).
    // We can fix one of the LDS dimensions to ROCWMMA_K (in this case the width),
    // and insert blocks of different heights (ROCWMMA_M, ROCWMMA_N) to use the space
    // without the need of extra padding.
    
    // Fragments of B must be transposed to fit this geometry,
    // and both fragments from A and B must accomodate LDS data layout.
    
    // Local Layout (LDS):
    //  Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
    // of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
    // B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
    // Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
    //
    //
    //                        _____________ROCWMMA_K_____________
    //                       |                                |
    //                       v                                v
    //                  (0,0) ----------------------------------->
    //          -->       -->  ______________    ...        ______
    //          |         |   |    |    |                  |      |
    //          |         |   |    |    |                  |      |
    //  Macro   |  ROCWMMA_M |   | C0 | C1 | C2               | Ck-1 |   A0
    //  Tile X  |         |   |    |    |                  |      |
    //          |         --> |___ |___ |____    ...       |______|
    //          |
    //          |                    ...  ...  ...  ...          AX-1
    //          -->
    //          -->       -->  ______________    ...        ______
    //          |         |   |    |    |                  |      |
    //          |         |   |    |    |                  |      |
    //  Macro   |  ROCWMMA_N |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
    //  Tile Y  |         |   |    |    |                  |      |
    //          |         --> |___ |___ |____    ...       |______|
    //          |
    //          |                    ...  ...  ...  ...        BY-1 (T)
    //          -->                                           (MacroTileX + MacroTileY - 1, ROCWMMA_K -1)
    //
    // TLDR: Take the Global Read fragments, transpose B and write the resulting frags into LDS
    // stacked on top of each other using ROCWMMA_K as common width.
    
    using LWFragA = fragment<matrix_a, 
                             T_BLOCK_X / rocwmma::Constants::AMDGCN_WAVE_SIZE * BLOCKS_X * ROCWMMA_M,
                             ROCWMMA_N, ROCWMMA_K, float64_t, row_major>;

    using LWFragB = fragment<matrix_a,  T_BLOCK_Y * BLOCKS_Y * ROCWMMA_N,
                             ROCWMMA_M, ROCWMMA_K, float64_t, row_major>;

    using LRFragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, row_major>;

    using LRFragB = fragment<matrix_a, ROCWMMA_N, ROCWMMA_M, ROCWMMA_K, float64_t, row_major>;


    // splitCountA - Number of work items to split the collaborative operation related to fragment A( Read / Write)
    
    // Ensure that splitCounts are the same on both sides of
    // global fetch and local writes to match fragment data locality.

    // IOCount - Total number of I/O operations needed for the entire block
    constexpr static auto splitCountA = std::min((uint32_t)rocwmma::GetIOTraits_t<typename std::decay<decltype(grBuffA)>::type>::IOCount,
                                                 (uint32_t)rocwmma::GetIOTraits_t<LWFragA>::IOCount);

    // splitCountB - Number of work items to split the collaborative operation related to fragment B( Read / Write)
    constexpr static auto splitCountB = std::min((uint32_t)rocwmma::GetIOTraits_t<typename std::decay<decltype(grBuffB)>::type>::IOCount,
                                                 (uint32_t)rocwmma::GetIOTraits_t<LWFragB>::IOCount);

    //Cooperatively load from global memory and split the work across waveCount
    globalReadCoopA<waveCount, splitCountA>(grBuffA, a + globalReadOffsetA, lda, waveIndex);

    globalReadCoopB<waveCount, splitCountB>(grBuffB, b + globalReadOffsetB, ldb, waveIndex);

    globalReadOffsetA += kStepOffsetA;
    globalReadOffsetB += kStepOffsetB;

    ///
    /// Setup LDS addressing
    /// This kernel will use 2 separate LDS blocks
    /// for pipelining in the accumulation loop
    ///
    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    constexpr static uint32_t LdsWidth = ROCWMMA_K;
    uint32_t sizeLds = (macroTileSizeC_x + macroTileSizeC_y) * LdsWidth;
    auto* ldsPtrLo = reinterpret_cast<float64_t*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    auto ldsWriteOffsetA = 0;
    auto ldsWriteOffsetB = macroTileSizeC_x * LdsWidth;

    auto ldsReadOffsetA = waveOffsetC_x * LdsWidth;
    auto ldsReadOffsetB = (macroTileSizeC_x + waveOffsetC_y) * LdsWidth;

    ///
    /// Write prefetch to local
    ///
    localWriteCoopA<waveCount, splitCountA>(ldsPtrLo + ldsWriteOffsetA, reinterpret_cast<LWFragA const&>(grBuffA), LdsWidth, waveIndex); 
    localWriteCoopB<waveCount, splitCountB>(ldsPtrLo + ldsWriteOffsetB, reinterpret_cast<LWFragB const&>(grBuffB), LdsWidth, waveIndex);

    ///
    /// Initialize accumulation frags
    ///
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, rocwmma::float64_t> fragsAcc[BLOCKS_X][BLOCKS_Y];
    fill(fragsAcc, static_cast<float64_t>(0));
 
    ///
    /// Synchronize waves and memory
    ///
    synchronize_workgroup();

    ///
    /// Accumulate A * B
    ///
     for(auto currentK = ROCWMMA_K; currentK < k; currentK += ROCWMMA_K)
    {
        fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> fragsA[BLOCKS_X];
        fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, row_major> fragsB[BLOCKS_Y];
 
        // Local read mfma frags
        localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, LdsWidth);
        localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, LdsWidth);

        // Start fetching next round of frags
        globalReadCoopA<waveCount, splitCountA>(grBuffA, a + globalReadOffsetA, lda, waveIndex);

        globalReadCoopB<waveCount, splitCountB>(grBuffB, b + globalReadOffsetB, ldb, waveIndex);
    
        // Advance offsets to next k step
        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        // accum(A * B)
        mfma(fragsAcc, fragsA, fragsB, fragsAcc);

        ///
        /// Write prefetch to local
        ///
        localWriteCoopA<waveCount, splitCountA>(ldsPtrHi + ldsWriteOffsetA, reinterpret_cast<LWFragA const&>(grBuffA), LdsWidth, waveIndex);
        localWriteCoopB<waveCount, splitCountB>(ldsPtrHi + ldsWriteOffsetB, reinterpret_cast<LWFragA const&>(grBuffB), LdsWidth, waveIndex);

        // Make sure that all waves have finished reading / writing to lds.
        synchronize_workgroup();

        // Swap Lds buffers
        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }

    ///
    /// Start loading C
    ///

    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> fragsC[BLOCKS_X][BLOCKS_Y];
    globalReadC(fragsC, c + globalReadOffsetC, ldc);

    ///
    /// Clean up tail A * B
    ///
    fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> fragsA[BLOCKS_X];
    fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, row_major> fragsB[BLOCKS_Y];

    // Local read mfma frags
    localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, LdsWidth);
    localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, LdsWidth);
    mfma(fragsAcc, fragsA, fragsB, fragsAcc);
  
    ///
    /// D = alpha * accum + beta * C
    ///
    fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float64_t, col_major> fragsD[BLOCKS_X][BLOCKS_Y];
    uniformFma(fragsD, alpha, fragsAcc, beta, fragsC);
    globalWriteD(d + globalWriteOffsetD, fragsD, ldd);
}

__host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float32_t alpha, float32_t beta)
{
    // Bounds check
    if((m < (ROCWMMA_M * T_BLOCK_X * BLOCKS_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y * BLOCKS_Y) || k < ROCWMMA_K)
       || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    // LytA_LytB_LytC = N_T_N_N
    int lda = m;
    int ldb = n;
    int ldc = m;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<float64_t> matrixA(m * k);
    std::vector<float64_t> matrixB(k * n);
    std::vector<float64_t> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float64_t> matrixD(m * n, std::numeric_limits<float64_t>::signaling_NaN());

    fill<float64_t, col_major>(matrixA.data(), m, k);
    fill<float64_t, row_major>(matrixB.data(), k, n);
    fill<float64_t, col_major>(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    float64_t* d_a;
    float64_t* d_b;
    float64_t* d_c;
    float64_t* d_d;

    const size_t bytesA = matrixA.size() * sizeof(float64_t);
    const size_t bytesB = matrixB.size() * sizeof(float64_t);
    const size_t bytesC = matrixC.size() * sizeof(float64_t);
    const size_t bytesD = matrixD.size() * sizeof(float64_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * BLOCKS_X * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceilDiv(n, ROCWMMA_N * BLOCKS_Y * T_BLOCK_Y));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << " gridDim " << gridDim.x << " " << gridDim.y << std::endl;
    std::cout << " blockdim " << blockDim.x << " " << blockDim.y << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
    
    // Uses 2 lds blocks for prefetch loop (A and B)
    int ldsusage = 2 * sizeof(float64_t) * 
                   (T_BLOCK_X / WAVE_SIZE * BLOCKS_X * ROCWMMA_M
                   + T_BLOCK_Y * BLOCKS_Y * ROCWMMA_N)
                   * ROCWMMA_K;

    hipExtLaunchKernelGGL(gemm_rocwmma_d,
                          gridDim,
                          blockDim,
                          ldsusage, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_d,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2*mnk
    auto gFlops       = 2.0 * static_cast<double>(m * n * k) * 1.0e-9;
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs);

    // Echo performance
    std::cout << "BlocksX, BlocksY, "
              << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << BLOCKS_X << ", " <<  BLOCKS_Y << ", " << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float64_t> matrixD_ref(m * n, std::numeric_limits<float64_t>::signaling_NaN());
    gemm_cpu_host(m,
               n,
               k,
               matrixA.data(),
               matrixB.data(),
               matrixC.data(),
               matrixD_ref.data(),
               lda,
               ldb,
               ldc,
               ldd,
               alpha,
               beta);
    
    compareEqual<float64_t>(matrixD.data(), matrixD_ref.data(), m * n);
             
    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    // Test for f64 device r
    if(!isF64Supported())
    {
        std::cout << "f64 dgemm not supported on this device" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    gemm_test(4096, 4096, 2048, 2, 2);
    return 0;
}
