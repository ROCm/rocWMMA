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

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Common.h"
#include "Utils.h"
#include "WMMA.h"
#include <functional>

// Training pass direction
enum class DlrmDirection_t : bool
{
    Forward,
    Backward
};

// Host forwards DLRM validation
__host__ void dlrmDotFwdCPU(float16_t* input, float16_t* output, uint32_t m, uint32_t k, uint32_t b)
{
    auto batchOffset = m * k;
    uint outputIdx   = 0;
    for(int t = 0; t < b; t++)
    {
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < k; j++)
            {
                float accum = 0.0f;
                for(int h = 0; h < k; h++)
                {
                    accum += static_cast<float>(input[t * batchOffset + i * k + h])
                             * static_cast<float>(input[t * batchOffset + j * k + h]);
                }
                // Copy MLP to output
                if(i == 0)
                {
                    output[outputIdx] = input[t * batchOffset + j];
                    outputIdx++;
                }
                if(j < i)
                {
                    output[outputIdx] = static_cast<float16_t>(accum);
                    outputIdx++;
                }
            }
        }
    }
}

// Host backwards DLRM validation
__host__ void dlrmDotBwdCPU(float16_t* input,
                            float16_t* upstreamGrad,
                            float16_t* bottomMlpGrad,
                            float16_t* output,
                            uint32_t   m,
                            uint32_t   k,
                            uint32_t   b)
{
    auto batchOffset = m * k;
    auto trilSize    = ((m * (m - 1)) / 2) + k;
    for(int t = 0; t < b; t++)
    {
        // Copy bottom MLP grad
        for(int j = 0; j < k; j++)
        {
            bottomMlpGrad[t * k + j] = upstreamGrad[t * trilSize + j];
        }

        // Remake tril
        float16_t temp[m * m];
        uint32_t  tempIdx = t * trilSize + k;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                if(i == j)
                {
                    temp[i * m + j] = 0;
                }
                else
                {
                    temp[i * m + j] = upstreamGrad[tempIdx];
                    temp[j * m + i] = upstreamGrad[tempIdx];
                    tempIdx++;
                }
            }
        }

        // Perform reverse bmm
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < k; j++)
            {
                float accum = 0.0f;
                for(int h = 0; h < m; h++)
                {
                    accum += static_cast<float>(temp[i * m + h])
                             * static_cast<float>(input[t * batchOffset + h * k + j]);
                }
                output[t * batchOffset + i * k + j] = static_cast<float16_t>(accum);
            }
        }
    }
}

// Supports WMMA fragment sizes (TILE_DIM) of
// : 16 x 16
// : 32 x 32
constexpr static const int TILE_DIM = 16;

// Thread block
// : T_BLOCK_X must be multiple of AMDGCN_WAVE_SIZE (64).
// Note: Each wave will compute one TILE_DIM x TILE_DIM output block
// Note: Workgroup will compute
//  T_BLOCK_X / AMDGCN_WAVE_SIZE output blocks
constexpr static const int T_BLOCK_X = 128;

// The following device kernel is a naive implementation
// of the forward-pass interaction dot layer in the DLRM
// architecture. Each wave will compute one TILE_DIM x TILE_DIM
// output block of the dot product, generalized as:
// D[b] = A[b] x transpose(A[b]) for B batches
//
// In this simplified example, we assume:
// : A is in row-major format            (M x K x B)
// : transpose(A) is in col-major format (K x M x B)
// : D is in row-major format            (M x M x B)
// : No LDS required
//
// This device kernel also handles concatenating the lower triangular
// indexing of D to the bottom MLP output to create the interaction dot output.
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level BMM computation, and is not optimized.
__global__ void dlrmDotFwd(const float16_t* __restrict input,
                           float16_t* __restrict output,
                           float* acc,
                           uint   m,
                           uint   k,
                           uint   b,
                           uint   inputBatchOffset,
                           uint   outputBatchOffset,
                           uint   accBatchOffset)
{
    using MappingA   = MappingUtil<TILE_DIM, TILE_DIM, float16_t, row_major>;
    using MappingB   = MappingUtil<TILE_DIM, TILE_DIM, float16_t, col_major>;
    using MappingC   = MappingUtil<TILE_DIM, TILE_DIM, float16_t, row_major>;
    using MappingAcc = MappingUtil<TILE_DIM, TILE_DIM, float, row_major>;

    using FragA   = wmma::fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, float16_t, row_major>;
    using FragB   = wmma::fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, float16_t, col_major>;
    using FragAcc = wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>;

    // Copy bottom MLP to output
    // Threads with a global index < k are responsible for copying MLP data
    auto globalThreadCoord = blockIdx.x * blockDim.x + threadIdx.x;
    auto count             = k >> Log2<T_BLOCK_X>::value;
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
        wmma::fill_fragment(fragAcc, static_cast<float>(0));

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
            wmma::synchronize_workgroup();

            auto fragA = FragA();
            auto fragB = FragB();

            // Load and multiply
            wmma::load_matrix_sync(fragA, addrA, k);
            wmma::load_matrix_sync(fragB, addrB, k);
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

            addrA += incrA;
            addrB += incrB;
        }
        wmma::synchronize_workgroup();

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
                    = float16_t(acc[accBatchOffset * blockIdx.z + globalRowIdx * m + globalColIdx]);
            }
        }
    }
}

// The following device kernel is a navie implementation of
// matrix reconstruction from a lower triangular indexing (tril) input.
//
// Note: This is a simplified implementation, and is not optimized
__global__ void trilReconstruct(const float16_t* __restrict upstreamGrad,
                                float16_t* __restrict acc,
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

// The following device kernel is a naive implementation
// of the backwards-pass interaction dot layer in the DLRM
// architecture. Each wave will compute one TILE_DIM x TILE_DIM
// output block of the dot product, generalized as:
// D[b] = reconstructedTril[b] x input[b] for B batches
//
// In this simplified example, we assume:
// : reconstructedTril is in row-major format (M x K x B)
// : input is in col-major format             (M x M x B)
// : D is in row-major format                 (M x K x B)
// : No LDS required
//
// This device kernel also handles copying the bottom MLP gradient.
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level BMM computation, and is not optimized.
__global__ void dlrmDotBwd(const float16_t* __restrict input,
                           const float16_t* __restrict upstreamGrad,
                           float16_t* __restrict grad,
                           float16_t* __restrict bottomMlpGrad,
                           float16_t* __restrict acc,
                           uint m,
                           uint k,
                           uint b,
                           uint inputBatchOffset,
                           uint upstreamBatchOffset,
                           uint accBatchOffset)
{
    using TileMapping = MappingUtil<TILE_DIM, TILE_DIM, float16_t, row_major>;

    using FragA   = wmma::fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, float16_t, row_major>;
    using FragB   = wmma::fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, float16_t, row_major>;
    using FragC   = wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float16_t>;
    using FragAcc = wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>;

    // Copy bottom MLP grad
    // Threads with a global index < k are responsible for copying MLP data
    auto globalThreadCoord = blockIdx.x * blockDim.x + threadIdx.x;
    auto count             = k >> Log2<T_BLOCK_X>::value;
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
    auto matrixCoordC = TileMapping::matrixCoord();

    // Target output gradient block to perform reverse bmm
    if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < k)
    {
        // Initialize accumulator
        auto fragAcc = FragAcc();
        wmma::fill_fragment(fragAcc, static_cast<float>(0));

        // Setup starting addresses
        auto* accWithOffset   = acc + accBatchOffset * blockIdx.z;
        auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
        auto* addrA           = TileMapping::dataCoord(
                      accWithOffset, m, std::make_pair(std::get<0>(matrixCoordC), 0));
        auto* addrB = TileMapping::dataCoord(
            inputWithOffset, k, std::make_pair(0, std::get<1>(matrixCoordC)));

        // Setup address increments.
        // A steps BlockK through m x m
        // B steps BlockK through m x k
        auto incrA = TileMapping::dataOffset(m, std::make_pair(0, TILE_DIM));
        auto incrB = TileMapping::dataOffset(k, std::make_pair(TILE_DIM, 0));

        auto count = m / TILE_DIM;
        for(int i = 0; i < count; i++)
        {
            auto fragA = FragA();
            auto fragB = FragB();

            // Load and multiply
            wmma::load_matrix_sync(fragA, addrA, m);
            wmma::load_matrix_sync(fragB, addrB, k);
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

            addrA += incrA;
            addrB += incrB;
        }

        // Output address
        auto* gradWithOffset = grad + inputBatchOffset * blockIdx.z;
        auto* addrGrad       = TileMapping::dataCoord(gradWithOffset, k, matrixCoordC);

        // Store accumulator fragment to output gradient
        auto fragC = FragC();

#pragma unroll
        for(int i = 0; i < fragC.num_elements; i++)
        {
            fragC.x[i] = float16_t(fragAcc.x[i]);
        }

        // Store the output
        wmma::store_matrix_sync(addrGrad, fragC, k, wmma::mem_row_major);
    }
}

__host__ void dlrm_test(uint32_t m, uint32_t k, uint32_t b, DlrmDirection_t passDirection)
{
    // Allocate and initialize host matrices
    std::vector<float16_t> h_input, h_output, h_upstreamGrad, h_grad, h_bottomMlpGrad;

    h_input.resize(m * k * b);

    fill<float16_t>(h_input.data(), m, k, b);

    const size_t trilSize = ((m * (m - 1)) / 2) + k;
    if(passDirection == DlrmDirection_t::Forward)
    {
        h_output.resize(trilSize * b);
    }
    else
    {
        h_upstreamGrad.resize(trilSize * b);
        h_grad.resize(m * k * b);
        h_bottomMlpGrad.resize(k * b);

        fill<float16_t>(h_upstreamGrad.data(), 1, trilSize, b);
    }

    // Allocate and copy device memory
    float16_t *d_input, *d_output, *d_upstreamGrad, *d_grad, *d_bottomMlpGrad, *d_accBwd;
    float*     d_accFwd;

    const size_t inputBytes         = h_input.size() * sizeof(float16_t);
    const size_t outputBytes        = h_output.size() * sizeof(float16_t);
    const size_t accFwdBytes        = m * m * b * sizeof(float32_t);
    const size_t accBwdBytes        = m * m * b * sizeof(float16_t);
    const size_t upstreamGradBytes  = h_upstreamGrad.size() * sizeof(float16_t);
    const size_t gradBytes          = h_grad.size() * sizeof(float16_t);
    const size_t bottomMlpGradBytes = h_bottomMlpGrad.size() * sizeof(float16_t);

    CHECK_HIP_ERROR(hipMalloc(&d_input, inputBytes));

    if(passDirection == DlrmDirection_t::Forward)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_output, outputBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_accFwd, accFwdBytes));

        CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputBytes, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMalloc(&d_upstreamGrad, upstreamGradBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_grad, gradBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_bottomMlpGrad, bottomMlpGradBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_accBwd, accBwdBytes));

        CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputBytes, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            d_upstreamGrad, h_upstreamGrad.data(), upstreamGradBytes, hipMemcpyHostToDevice));
    }

    std::function<void()> dlrmKernel;

    if(passDirection == DlrmDirection_t::Forward)
    {
        dlrmKernel = [d_input, d_output, d_accFwd, m, k, b]() {
            auto gridDim = dim3(
                ceilDiv(m, TILE_DIM * T_BLOCK_X / AMDGCN_WAVE_SIZE), ceilDiv(m, TILE_DIM), b);
            auto blockDim = dim3(T_BLOCK_X);

            uint inputBatchOffset  = m * k;
            uint outputBatchOffset = ((m * (m - 1)) / 2) + k;
            uint accBatchOffset    = m * m;

            hipExtLaunchKernelGGL((dlrmDotFwd),
                                  gridDim,
                                  blockDim,
                                  0, // sharedMemBytes
                                  0, // stream
                                  nullptr, // event start
                                  nullptr, // event stop
                                  0, // flags
                                  d_input,
                                  d_output,
                                  d_accFwd,
                                  m,
                                  k,
                                  b,
                                  inputBatchOffset,
                                  outputBatchOffset,
                                  accBatchOffset);
        };
    }
    else
    {
        dlrmKernel = [d_input, d_upstreamGrad, d_grad, d_bottomMlpGrad, d_accBwd, m, k, b]() {
            auto gridDim  = dim3(ceilDiv(m * m, T_BLOCK_X), 1, b);
            auto blockDim = dim3(T_BLOCK_X);

            uint inputBatchOffset    = m * k;
            uint upstreamBatchOffset = ((m * (m - 1)) / 2) + k;
            uint accBatchOffset      = m * m;

            hipEvent_t syncEvent;
            CHECK_HIP_ERROR(hipEventCreate(&syncEvent));
            hipExtLaunchKernelGGL((trilReconstruct),
                                  gridDim,
                                  blockDim,
                                  0, // sharedMemBytes
                                  0, // stream
                                  nullptr, // event start
                                  nullptr, // event stop
                                  0, // flags
                                  d_upstreamGrad,
                                  d_accBwd,
                                  m,
                                  k,
                                  b,
                                  upstreamBatchOffset,
                                  accBatchOffset);
            CHECK_HIP_ERROR(hipEventRecord(syncEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

            gridDim = dim3(
                ceilDiv(m, TILE_DIM * T_BLOCK_X / AMDGCN_WAVE_SIZE), ceilDiv(k, TILE_DIM), b);

            hipExtLaunchKernelGGL((dlrmDotBwd),
                                  gridDim,
                                  blockDim,
                                  0, // sharedMemBytes
                                  0, // stream
                                  nullptr, // event start
                                  nullptr, // event stop
                                  0, // flags
                                  d_input,
                                  d_upstreamGrad,
                                  d_grad,
                                  d_bottomMlpGrad,
                                  d_accBwd,
                                  m,
                                  k,
                                  b,
                                  inputBatchOffset,
                                  upstreamBatchOffset,
                                  accBatchOffset);
        };
    }

    std::cout << "Launching "
              << (passDirection == DlrmDirection_t::Forward ? "forwards" : "backwards")
              << " DLRM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    dlrmKernel();
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto timeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // DLRM flops converge to 2*mmkb
    auto gFlops       = 2.0 * static_cast<double>(m * m * k * b) * 1.0e-9;
    auto gFlopsPerSec = gFlops / static_cast<double>(timeMs) * 1.0e3;

    // Echo performance
    std::cout << "TileSize, "
              << "MatM, MatK, Batches, "
              << "elapsedMs, GFlops, GFlops/s" << std::endl;

    std::cout << TILE_DIM << ", " << m << ", " << k << ", " << b << ", " << timeMs << ", " << gFlops
              << ", " << gFlopsPerSec << std::endl;

    std::cout << "Validating result with reference..." << std::endl;

    if(passDirection == DlrmDirection_t::Forward)
    {
        CHECK_HIP_ERROR(hipMemcpy(h_output.data(), d_output, outputBytes, hipMemcpyDeviceToHost));

        std::vector<float16_t> outputRef;
        outputRef.resize(h_output.size());

        dlrmDotFwdCPU(h_input.data(), outputRef.data(), m, k, b);

        compareEqual<float16_t>(h_output.data(), outputRef.data(), h_output.size(), 1.0);
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(h_grad.data(), d_grad, gradBytes, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_bottomMlpGrad.data(), d_bottomMlpGrad, bottomMlpGradBytes, hipMemcpyDeviceToHost));

        std::vector<float16_t> gradRef, bottomMlpGradRef;
        gradRef.resize(h_grad.size());
        bottomMlpGradRef.resize(h_bottomMlpGrad.size());

        dlrmDotBwdCPU(h_input.data(),
                      h_upstreamGrad.data(),
                      bottomMlpGradRef.data(),
                      gradRef.data(),
                      m,
                      k,
                      b);

        compareEqual<float16_t>(h_grad.data(), gradRef.data(), h_grad.size(), 1.0);
        compareEqual<float16_t>(
            h_bottomMlpGrad.data(), bottomMlpGradRef.data(), h_bottomMlpGrad.size(), 1.0);
    }
}

int main()
{
    dlrm_test(32, 128, 64, DlrmDirection_t::Forward);
    dlrm_test(32, 128, 64, DlrmDirection_t::Backward);
    return 0;
}
