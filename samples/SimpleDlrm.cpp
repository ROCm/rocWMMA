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

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Common.h"
#include "WMMA.h"
#include <functional>
#include <sys/types.h>

struct __align__(8) half4
{
    half2 vals[2];
};

__device__ inline void store(__half* dst, float* src)
{
    *dst = __float2half(*src);
}

__device__ inline void store(__half* dst, const float src)
{
    *dst = __float2half(src);
}

template <uint x>
struct Log2
{
    static constexpr uint value = 1 + Log2<x / 2>::value;
};

template <>
struct Log2<1>
{
    static constexpr uint value = 0;
};

enum : uint32_t
{
    // Data size parameters
    numRows   = 32,
    numCols   = 64,
    batchSize = 2,

    // Shared kernel template parameters
    WARP_SIZE             = AMDGCN_WAVE_SIZE,
    WARP_SIZE_LOG2        = Log2<WARP_SIZE>::value,
    TILE_DIM              = 16,
    TILE_DIM_LOG2         = Log2<TILE_DIM>::value,
    PAD                   = 0,
    MEM_SKEW_SIZE         = 8,
    WARPS_PER_THREADBLOCK = 128 / WARP_SIZE,
    THREADBLOCK_SIZE      = WARPS_PER_THREADBLOCK * WARP_SIZE,

    // Forward kernel template parameters
    M_BLOCKS        = numRows / TILE_DIM,
    K_BLOCKS        = numCols / TILE_DIM,
    SMEM_STRIDE     = K_BLOCKS * 16 + 8,
    SMEM_STRIDE_ACC = M_BLOCKS * 16 + 8,

    // Backward kernel template parameters
    ROW_TILES_PER_STEP = 32 / TILE_DIM,
    COL_TILES_PER_STEP = 1
};

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT* mat, uint32_t numRows, uint32_t numCols, uint32_t batchSize)
{
    auto batchOffset = numRows * numCols;
    for(int k = 0; k < batchSize; ++k)
    {
        for(int i = 0; i < numRows; ++i)
        {
            for(int j = 0; j < numCols; ++j)
            {
                // Random values normalized such that output is between 0 and 1
                auto value = __float2half(static_cast<float>(rand() / numCols)
                                          / static_cast<float>(RAND_MAX));
                mat[k * batchOffset + i * numCols + j] = static_cast<DataT>(value);
            }
        }
    }
}

__host__ void bmmTrillPadCPU(
    float16_t* input, float16_t* output, uint32_t numRows, uint32_t numCols, uint32_t batchSize)
{
    auto batchOffset = numRows * numCols;
    uint outputIdx   = 0;
    for(int k = 0; k < batchSize; k++)
    {
        for(int i = 0; i < numRows; i++)
        {
            for(int j = 0; j < numCols; j++)
            {
                float16_t accum = 0.0f;
                for(int h = 0; h < numCols; h++)
                {
                    accum += static_cast<float16_t>(input[k * batchOffset + i * numCols + h])
                             * static_cast<float16_t>(input[k * batchOffset + j * numCols + h]);
                }
                // Copy MLP to output
                if(i == 0)
                {
                    //output[k * batchOffset + j] = input[k * batchOffset + j];
                    output[outputIdx] = input[k * batchOffset + j];
                    outputIdx++;
                }
                if(j < i)
                {
                    //output[k * batchOffset + outputIdx] = accum;
                    output[outputIdx] = accum;
                    outputIdx++;
                }
            }
        }
    }
}

/*
* deconcat bottom_mlp_grad = grad[0:numCols]
           interaction_grad = gard[numCols:]
*
* deflatten into tril (trilbwdkernel does this)
*
* copy bottom tril to top (flip i & j)
*   (pad to 32)
*
* reverse mm interaction_grad_2d * input = output_grad
             [27x27]               [27x128]
             (bmmBwdkernel does this)
*
*/

__host__ void bmmBwdCPU(float16_t* input,
                        float16_t* upstreamGrad,
                        float16_t* bottomMlpGrad,
                        float16_t* output,
                        uint32_t   numRows,
                        uint32_t   numCols,
                        uint32_t   batchSize)
{
    auto batchOffset = numRows * numCols;
    auto trilSize    = ((numRows * (numRows - 1)) / 2) + numCols;
    for(int k = 0; k < batchSize; k++)
    {
        // Copy bottom MLP grad
        for(int j = 0; j < numCols; j++)
        {
            bottomMlpGrad[k * numCols + j] = upstreamGrad[k * trilSize + j];
        }

        // Remake tril
        float16_t temp[numRows * numRows];
        uint32_t  tempIdx = k * trilSize + numCols;
        for(int i = 0; i < numRows; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                if(i == j)
                {
                    temp[i * numRows + j] = 0;
                }
                else
                {
                    temp[i * numRows + j] = upstreamGrad[tempIdx];
                    temp[j * numRows + i] = upstreamGrad[tempIdx];
                    tempIdx++;
                }
            }
        }

        // Perform reverse bmm
        for(int i = 0; i < numRows; i++)
        {
            for(int j = 0; j < numCols; j++)
            {
                float16_t accum = 0.0f;
                for(int h = 0; h < numRows; h++)
                {
                    accum += static_cast<float16_t>(temp[i * numRows + h])
                             * static_cast<float16_t>(input[k * batchOffset + h * numCols + j]);
                }
                output[k * batchOffset + i * numCols + j] = accum;
            }
        }
    }
}

__device__ inline void bmmTrilPadFwdKernel(half* shmem,
                                           half* gmem_output,
                                           uint  numRows,
                                           uint  numCols,
                                           uint  smemRowsPerWarp,
                                           uint  outputSize,
                                           uint  numColSteps,
                                           uint  PAD,
                                           int   lane_id)
{
    wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[M_BLOCKS][M_BLOCKS];

    for(int i = 0; i < M_BLOCKS; i++)
    {
        for(int j = 0; j < M_BLOCKS; j++)
        {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for(int k_step = 0; k_step < numColSteps; k_step++)
    {
        wmma::fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, row_major> a[M_BLOCKS];
        wmma::fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, col_major> b[M_BLOCKS];
        for(int j = 0; j < M_BLOCKS; j++)
        {
            int         base_row = (j < M_BLOCKS - 1) ? j * 16 : smemRowsPerWarp - 16;
            const half* tile_ptr
                = reinterpret_cast<half*>(shmem) + (base_row * SMEM_STRIDE + k_step * 16);
            wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
            wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
        }
        for(int i = 0; i < M_BLOCKS; i++)
        {
            for(int j = 0; j < M_BLOCKS; j++)
            {
                wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }
    float* shmem_store = reinterpret_cast<float*>(shmem);
    for(int i = 0; i < M_BLOCKS; i++)
    {
        for(int j = 0; j < M_BLOCKS; j++)
        {
            float* tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
            wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC, wmma::mem_row_major);
        }
    }

    half* gmem_interact_output = gmem_output + numCols;
    int   lastRowBlockOffset   = M_BLOCKS * 16 - smemRowsPerWarp;
    int   srcLine              = 0;
    for(int i = 0; i < numRows; ++i, ++srcLine)
    {
        if(i == ((M_BLOCKS - 1) * 16))
        {
            srcLine += lastRowBlockOffset;
        }
        if(lane_id < i)
        {
            uint offset = (i * (i - 1)) >> 1;
            store(gmem_interact_output + offset + lane_id,
                  shmem_store + srcLine * SMEM_STRIDE_ACC + lane_id);
        }
    }

    // Padding
    if(lane_id < PAD)
    {
        store(gmem_output + lane_id + outputSize - 1, 0.0f);
    }
}

__device__ inline void trilBwdKernel(half* smem_in,
                                     half* smem_temp,
                                     half  zero,
                                     uint  numRows,
                                     uint  numRowsAfterPadding,
                                     uint  interactionUgrad2DStride,
                                     uint  lane_id)
{
    if(lane_id < numRowsAfterPadding)
    {
        uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
        uint ugrad_offset_1   = lane_id * interactionUgrad2DStride;
        for(uint row = 0; row < numRows; row++)
        {
            half ugrad_val = zero;
            if(row < lane_id && lane_id < numRows)
            {
                ugrad_val                       = smem_in[ugrad_flat_index + row];
                smem_temp[ugrad_offset_1 + row] = ugrad_val;
            }
            if(row <= lane_id && lane_id < numRowsAfterPadding)
            {
                smem_temp[row * interactionUgrad2DStride + lane_id] = ugrad_val;
            }
        }
        for(uint row = numRows; row < numRowsAfterPadding; row++)
        {
            smem_temp[row * interactionUgrad2DStride + lane_id] = zero;
        }
    }
}

__device__ inline void bmmBwdKernel(half*  smem_in,
                                    half*  smem_temp,
                                    float* smem_out,
                                    half*  gmem_grad,
                                    uint   numRows,
                                    uint   numCols,
                                    uint   numColSteps,
                                    uint   inputStride,
                                    uint   interactionUgrad2DStride,
                                    uint   lane_id)
{
    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major>
        a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
    for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
    {
        for(uint j = 0; j < ROW_TILES_PER_STEP; j++)
        {
            const half* tile_ptr
                = smem_temp + ((i * interactionUgrad2DStride + j) << TILE_DIM_LOG2);
            wmma::load_matrix_sync(a[i][j], tile_ptr, interactionUgrad2DStride);
        }
    }

    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[ROW_TILES_PER_STEP];
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major>
        b[ROW_TILES_PER_STEP];
    for(int col_step = 0; col_step < numColSteps; col_step++)
    {
        for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
        {
            const half* tile_ptr = smem_in + ((i * inputStride + col_step) << TILE_DIM_LOG2);
            wmma::fill_fragment(acc[i], 0.0f);
            wmma::load_matrix_sync(b[i], tile_ptr, inputStride);
        }
        for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
        {
            for(uint j = 0; j < ROW_TILES_PER_STEP; j++)
            {
                wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
            }
        }
        for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
        {
            float* tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
            wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, wmma::mem_row_major);
        }

        __syncthreads();

        uint gmem_grad_col = (col_step << TILE_DIM_LOG2) + lane_id;
        if(gmem_grad_col < numCols)
        {
            for(uint i = 0; i < numRows; i++)
            {
                store(&gmem_grad[i * numCols + gmem_grad_col],
                      &smem_out[(i << TILE_DIM_LOG2) + lane_id]);
            }
        }
    }
}

__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractFwdKernel(const __half* __restrict input,
                                   __half* __restrict output,
                                   uint batchSize,
                                   uint numRows,
                                   uint numCols,
                                   uint numRowsAfterPadding,
                                   uint numColsAfterPadding,
                                   uint smemElemsPerWarp,
                                   uint smemRowsPerWarp,
                                   uint outputSize,
                                   uint numRowSteps,
                                   uint numColSteps,
                                   uint PAD)
{
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG2);
    int  sample_id = blockIdx.x * WARPS_PER_THREADBLOCK + warp_id;
    if(sample_id >= batchSize)
    {
        return;
    }
    int lane_id = threadIdx.x & (WARP_SIZE - 1);

    HIP_DYNAMIC_SHARED(half, shmem_dynamic_half)
    half* shmem = shmem_dynamic_half + (warp_id * smemElemsPerWarp);

    const half* sample_input = input + numRows * numCols * sample_id;
    if(lane_id < (numCols >> 2))
    {
        for(int i = 0; i < numRows; ++i, sample_input += numCols)
        {
            ((float2*)(shmem + i * SMEM_STRIDE))[lane_id] = ((float2*)sample_input)[lane_id];
        }
    }

    uint idx = lane_id + numCols;
    if(idx < numColsAfterPadding)
    {
        for(int i = 0; i < numRows; ++i)
        {
            (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
        }
    }

    half4 zeros;
    zeros.vals[0] = __float2half2_rn(0);
    zeros.vals[1] = __float2half2_rn(0);

    if(lane_id < (numColsAfterPadding >> 2))
    {
        for(int i = numRows; i < numRowsAfterPadding; i++)
        {
            ((half4*)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
        }
    }

    __syncthreads();

    half* gmem_output = output + outputSize * sample_id;
    if(lane_id < (numCols >> 2))
    {
        ((float2*)gmem_output)[lane_id] = ((float2*)shmem)[lane_id];
    }

    bmmTrilPadFwdKernel(shmem,
                        gmem_output,
                        numRows,
                        numCols,
                        smemRowsPerWarp,
                        outputSize,
                        numColSteps,
                        PAD,
                        lane_id);
}

__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernel(const __half* __restrict input,
                                   const __half* __restrict upstream_grad,
                                   half* __restrict grad,
                                   half* __restrict bottom_mlp_grad,
                                   uint batchSize,
                                   uint numRows,
                                   uint numCols,
                                   uint numRowsAfterPadding,
                                   uint numColsAfterPadding,
                                   uint sampleSize,
                                   uint interactionUgradSize,
                                   uint interactionUgradSizeWithPadding,
                                   uint interactionUgrad2DSizeElems,
                                   uint interactionUgrad2DStride,
                                   uint inputSizeElems,
                                   uint inputStride,
                                   uint numRowSteps,
                                   uint numColSteps,
                                   uint rowTilesPerStep,
                                   uint sharedMemPerWarpSizeByte)
{
    HIP_DYNAMIC_SHARED(half, shared_mem_half)
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG2);
    uint sample_id = blockIdx.x * WARPS_PER_THREADBLOCK + warp_id;
    if(sample_id >= batchSize)
    {
        return;
    }
    uint lane_id = threadIdx.x & (WARP_SIZE - 1);
    // ">> 1" to convert to half pointer
    uint smem_warp_offset = warp_id * (sharedMemPerWarpSizeByte >> 1);
    uint wmma_offset      = numRowsAfterPadding * numRowsAfterPadding; // matrix_a is row_major

    half*  smem_in   = shared_mem_half + smem_warp_offset + wmma_offset;
    half*  smem_temp = smem_in + inputSizeElems;
    float* smem_out  = reinterpret_cast<float*>(smem_temp);

    // Global memory pointers for the current sample
    // Input
    uint        gmem_input_sample_offset = sample_id * sampleSize;
    const half* gmem_input               = &input[gmem_input_sample_offset];

    // Interaction Gradient
    const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
    half*       gmem_grad               = &grad[gmem_grad_sample_offset];

    // Bottom MLP gradient
    half* gmem_mlp_grad = &bottom_mlp_grad[sample_id * numCols];

    // Upstream gradient vector
    uint        gmem_ugrad_sample_offset = sample_id * (numCols + interactionUgradSizeWithPadding);
    const half* gmem_ugrad               = &upstream_grad[gmem_ugrad_sample_offset];

    // Upstream gradient vector for interactions
    const half* gmem_ugrad_interactions = &gmem_ugrad[numCols];

    // upstream grad -> shared memory (place in input section temporarily)
    for(uint idx = lane_id; idx < (interactionUgradSize >> 3); idx += WARP_SIZE)
    {
        ((float4*)smem_in)[idx] = ((float4*)gmem_ugrad_interactions)[idx];
    }
    uint offset = (interactionUgradSize >> 3) << 3;
    for(uint idx = lane_id + offset; idx < interactionUgradSize; idx += WARP_SIZE)
    {
        smem_in[idx] = gmem_ugrad_interactions[idx];
    }

    __syncthreads();

    // Form the 2D ugrad matrix.
    trilBwdKernel(smem_in,
                  smem_temp,
                  __float2half(0),
                  numRows,
                  numRowsAfterPadding,
                  interactionUgrad2DStride,
                  lane_id);

    __syncthreads();

    // Input -> Shared Memory

    if(lane_id < (numCols >> 2))
    {
        for(uint row = 0; row < numRows; row++)
        {
            half*       smem_row_ptr         = &smem_in[row * inputStride];
            const half* gmem_row_ptr         = &gmem_input[row * numCols];
            ((float2*)smem_row_ptr)[lane_id] = ((float2*)gmem_row_ptr)[lane_id];
        }
    }

    uint idx = lane_id + numCols;
    if(idx < numColsAfterPadding)
    {
        for(uint row = 0; row < numRows; row++)
        {
            half* smem_row_ptr = &smem_in[row * inputStride];
            smem_row_ptr[idx]  = __float2half(0);
        }
    }

    half4 zeros;
    zeros.vals[0] = __float2half2_rn(0);
    zeros.vals[1] = __float2half2_rn(0);

    if(lane_id < (numColsAfterPadding >> 2))
    {
        for(uint row = numRows; row < numRowsAfterPadding; row++)
        {
            half* smem_row_ptr              = &smem_in[row * inputStride];
            ((half4*)smem_row_ptr)[lane_id] = zeros;
        }
    }

    __syncthreads();

    bmmBwdKernel(smem_in,
                 smem_temp,
                 smem_out,
                 gmem_grad,
                 numRows,
                 numCols,
                 numColSteps,
                 inputStride,
                 interactionUgrad2DStride,
                 lane_id);

    if(lane_id < (numCols >> 2))
    {
        ((float2*)gmem_mlp_grad)[lane_id] = ((float2*)gmem_ugrad)[lane_id];
    }
}

__host__ void dlrm_test(bool isBwd)
{
    // Allocate and initialize host matrices
    std::vector<float16_t> h_input, h_output, h_upstreamGrad, h_grad, h_bottomMlpGrad;

    h_input.resize(numRows * numCols * batchSize);

    fill<float16_t>(h_input.data(), numRows, numCols, batchSize);

    const size_t trilSize = ((numRows * (numRows - 1)) / 2) + numCols;
    if(!isBwd)
    {
        h_output.resize(trilSize * batchSize);
    }
    else
    {
        h_upstreamGrad.resize(trilSize * batchSize);
        h_grad.resize(numRows * numCols * batchSize);
        h_bottomMlpGrad.resize(numCols * batchSize);

        fill<float16_t>(h_upstreamGrad.data(), 1, trilSize, batchSize);
    }

    // Allocate and copy device memory
    float16_t *d_input, *d_output, *d_upstreamGrad, *d_grad, *d_bottomMlpGrad;

    const size_t inputBytes         = h_input.size() * sizeof(float16_t);
    const size_t outputBytes        = h_output.size() * sizeof(float16_t);
    const size_t upstreamGradBytes  = h_upstreamGrad.size() * sizeof(float16_t);
    const size_t gradBytes          = h_grad.size() * sizeof(float16_t);
    const size_t bottomMlpGradBytes = h_bottomMlpGrad.size() * sizeof(float16_t);

    CHECK_HIP_ERROR(hipMalloc(&d_input, inputBytes));

    if(!isBwd)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_output, outputBytes));

        CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputBytes, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMalloc(&d_upstreamGrad, upstreamGradBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_grad, gradBytes));
        CHECK_HIP_ERROR(hipMalloc(&d_bottomMlpGrad, bottomMlpGradBytes));

        CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputBytes, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            d_upstreamGrad, h_upstreamGrad.data(), upstreamGradBytes, hipMemcpyHostToDevice));
    }

    std::function<void()> dlrmKernel;

    if(!isBwd)
    {
        // Forward kernel argument parameters
        // num tiles
        uint numRowTiles = ceilDiv(static_cast<uint>(numRows), static_cast<uint>(TILE_DIM));
        uint numColTiles = ceilDiv(static_cast<uint>(numCols), static_cast<uint>(TILE_DIM));

        // number of rows and columns after padding
        uint numRowsAfterPadding = numRowTiles << TILE_DIM_LOG2;
        uint numColsAfterPadding = numColTiles << TILE_DIM_LOG2;

        uint numRowSteps = numRowTiles / ROW_TILES_PER_STEP;
        uint numColSteps = numColTiles / COL_TILES_PER_STEP;

        // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
        const uint smemRowsPerWarp     = M_BLOCKS << 4;
        const uint smemElemsPerWarpMat = smemRowsPerWarp * SMEM_STRIDE;

        // output in FP32
        const uint smemElemsPerWarpAcc = M_BLOCKS * TILE_DIM * SMEM_STRIDE_ACC * 2;
        const uint smemElemsPerWarp    = (smemElemsPerWarpMat > smemElemsPerWarpAcc)
                                             ? smemElemsPerWarpMat
                                             : smemElemsPerWarpAcc;

        dlrmKernel = [d_input,
                      d_output,
                      numRowsAfterPadding,
                      numColsAfterPadding,
                      smemElemsPerWarp,
                      smemRowsPerWarp,
                      trilSize,
                      numRowSteps,
                      numColSteps]() {
            hipLaunchKernelGGL((dotBasedInteractFwdKernel),
                               dim3(ceilDiv(static_cast<uint>(batchSize),
                                            static_cast<uint>(WARPS_PER_THREADBLOCK))),
                               dim3(THREADBLOCK_SIZE),
                               WARPS_PER_THREADBLOCK * smemElemsPerWarp * sizeof(__half),
                               0,
                               (const __half*)d_input,
                               (half*)d_output,
                               batchSize,
                               numRows,
                               numCols,
                               numRowsAfterPadding,
                               numColsAfterPadding,
                               smemElemsPerWarp,
                               smemRowsPerWarp,
                               trilSize,
                               numRowSteps,
                               numColSteps,
                               PAD);
        };
    }
    else
    {
        // Backward kernel argument parameters
        const uint kWarpsPerBlockLog2 = Log2<WARPS_PER_THREADBLOCK>::value;
        const uint tileSizeLog2       = Log2<TILE_DIM>::value;

        uint inputDataBytes = sizeof(half);

        uint rowTilesPerStep = numRows > TILE_DIM ? ROW_TILES_PER_STEP : 1;

        // num tiles
        uint numRowTiles = (numRows + TILE_DIM - 1) >> tileSizeLog2;
        uint numColTiles = (numCols + TILE_DIM - 1) >> tileSizeLog2;

        // number of rows and columns after padding
        uint numRowsAfterPadding = numRowTiles << tileSizeLog2;
        uint numColsAfterPadding = numColTiles << tileSizeLog2;

        // 2D ugrad size and stride
        uint interactionUgrad2DStride    = numRowsAfterPadding + MEM_SKEW_SIZE;
        uint interactionUgrad2DSizeElems = numRowsAfterPadding * interactionUgrad2DStride;
        uint interactionUgrad2DSizeBytes = interactionUgrad2DSizeElems * inputDataBytes;

        // 1D ugrad size
        uint interactionUgradSize            = numRows * (numRows - 1) >> 1;
        uint interactionUgradSizeWithPadding = interactionUgradSize + PAD;

        // in_out place size and stride
        uint inputStride    = numColsAfterPadding + MEM_SKEW_SIZE;
        uint inputSizeElems = numRowsAfterPadding * inputStride;
        uint inputSizeBytes = inputSizeElems * inputDataBytes;

        // sample size
        uint sampleSize = numRows * numCols;

        // output size
        uint outputSizeElems = TILE_DIM * TILE_DIM * ROW_TILES_PER_STEP * COL_TILES_PER_STEP;
        uint outputSizeBytes = outputSizeElems * sizeof(float);

        // staging area size
        uint stagingAreaSizeBytes = outputSizeBytes > interactionUgrad2DSizeBytes
                                        ? outputSizeBytes
                                        : interactionUgrad2DSizeBytes;

        // Shared memory size
        uint wmmaSmemBytes            = numRowsAfterPadding * numRowsAfterPadding * inputDataBytes;
        uint sharedMemPerWarpSizeByte = inputSizeBytes + stagingAreaSizeBytes + wmmaSmemBytes;
        uint sharedMemSizeBytes       = WARPS_PER_THREADBLOCK * sharedMemPerWarpSizeByte;

        uint numBlocks   = (batchSize + WARPS_PER_THREADBLOCK - 1) >> kWarpsPerBlockLog2;
        uint numRowSteps = numRowTiles / rowTilesPerStep;
        uint numColSteps = numColTiles / COL_TILES_PER_STEP;

        dlrmKernel = [d_input,
                      d_upstreamGrad,
                      d_grad,
                      d_bottomMlpGrad,
                      numBlocks,
                      sharedMemSizeBytes,
                      numRowsAfterPadding,
                      numColsAfterPadding,
                      sampleSize,
                      interactionUgradSize,
                      interactionUgradSizeWithPadding,
                      interactionUgrad2DSizeElems,
                      interactionUgrad2DStride,
                      inputSizeElems,
                      inputStride,
                      numRowSteps,
                      numColSteps,
                      rowTilesPerStep,
                      sharedMemPerWarpSizeByte]() {
            hipLaunchKernelGGL((dotBasedInteractBwdKernel),
                               dim3(numBlocks),
                               dim3(THREADBLOCK_SIZE),
                               sharedMemSizeBytes,
                               0,
                               (const half*)d_input,
                               (const half*)d_upstreamGrad,
                               (half*)d_grad,
                               (half*)d_bottomMlpGrad,
                               batchSize,
                               numRows,
                               numCols,
                               numRowsAfterPadding,
                               numColsAfterPadding,
                               sampleSize,
                               interactionUgradSize,
                               interactionUgradSizeWithPadding,
                               interactionUgrad2DSizeElems,
                               interactionUgrad2DStride,
                               inputSizeElems,
                               inputStride,
                               numRowSteps,
                               numColSteps,
                               rowTilesPerStep,
                               sharedMemPerWarpSizeByte);
        };
    }

    std::cout << "Launching " << (isBwd ? "Bwd" : "Fwd") << " Dlrm kernel..." << std::endl;

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

    if(!isBwd)
    {
        CHECK_HIP_ERROR(hipMemcpy(h_output.data(), d_output, outputBytes, hipMemcpyDeviceToHost));

        std::vector<float16_t> outputRef;
        outputRef.resize(h_output.size());

        bmmTrillPadCPU(h_input.data(), outputRef.data(), numRows, numCols, batchSize);

        // for(int i = 0; i < outputRef.size(); i++)
        //     std::cout << "Host: " << outputRef[i] << ", Device: " << h_output[i] << '\n';

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

        bmmBwdCPU(h_input.data(),
                  h_upstreamGrad.data(),
                  bottomMlpGradRef.data(),
                  gradRef.data(),
                  numRows,
                  numCols,
                  batchSize);

        // for (int i = 0; i < h_grad.size(); i++)
        //     std::cout << "Host: " << gradRef[i] << ", Device: " << h_grad[i] << '\n';

        compareEqual<float16_t>(h_grad.data(), gradRef.data(), h_grad.size(), 1.0);

        // for (int i = 0; i < h_bottomMlpGrad.size(); i++)
        //     std::cout << "Host: " << bottomMlpGradRef[i] << ", Device: " << h_bottomMlpGrad[i] << '\n';

        compareEqual<float16_t>(
            h_bottomMlpGrad.data(), bottomMlpGradRef.data(), h_bottomMlpGrad.size(), 1.0);
    }
}

int main()
{
    dlrm_test(false);
    dlrm_test(true);
    return 0;
}
