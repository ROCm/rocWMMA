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

#include "WMMA.h"
#include <functional>
#include <sys/types.h>

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

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

__device__ inline void store(float* dst, float* src)
{
    *dst = *src;
}

__device__ inline void store(float* dst, const float src)
{
    *dst = src;
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
                mat[k * batchOffset + i * numRows + j] = static_cast<DataT>(value);
            }
        }
    }
}

// Element-wise comparison
__host__ void
    compareEqual(float16_t const* a, float16_t const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval;
    double max_relative_error = 0.0;

    for(int i = 0; i < size; i++)
    {
        auto valA           = a[i];
        auto valB           = b[i];
        auto relative_error = fabs(valA - valB) / (fabs(valA) + fabs(valB) + 1.0);

        if(relative_error > max_relative_error || relative_error != relative_error)
        {
            max_relative_error = relative_error;
        }
    }
    auto eps = std::numeric_limits<float16_t>::epsilon();
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAILED\n";
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << max_relative_error << std::endl;
}

__host__ void bmmTrillPadCPU(
    float16_t* input, float16_t* output, uint32_t numRows, uint32_t numCols, uint32_t batchSize)
{
    auto batchOffset = numRows * numCols;
    uint outputIdx   = 0;
    uint j;
    for(int k = 0; k < batchSize; k++)
    {
        for(int i = 0; i < numRows; i++)
        {
            for(j = 0; j < numCols; j++)
            {
                float16_t accum = 0.0f;
                for(int h = 0; h < numCols; h++)
                {
                    accum += static_cast<float16_t>(input[k * batchOffset + i * numRows + h])
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

template <uint TILE_DIM, uint M_BLOCKS, uint SMEM_STRIDE, uint SMEM_STRIDE_ACC>
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

template <uint TILE_DIM, uint ROW_TILES_PER_STEP, uint TILE_DIM_LOG_2>
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
                = smem_temp + ((i * interactionUgrad2DStride + j) << TILE_DIM_LOG_2);
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
            const half* tile_ptr = smem_in + ((i * inputStride + col_step) << TILE_DIM_LOG_2);
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

        uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
        if(gmem_grad_col < numCols)
        {
            for(uint i = 0; i < numRows; i++)
            {
                store(&gmem_grad[i * numCols + gmem_grad_col],
                      &smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
            }
        }
    }
}

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint M_BLOCKS,
          uint K_BLOCKS,
          uint SMEM_STRIDE,
          uint SMEM_STRIDE_ACC,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
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
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    int  sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
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

    bmmTrilPadFwdKernel<TILE_DIM, M_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC>(shmem,
                                                                          gmem_output,
                                                                          numRows,
                                                                          numCols,
                                                                          smemRowsPerWarp,
                                                                          outputSize,
                                                                          numColSteps,
                                                                          PAD,
                                                                          lane_id);
}

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
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
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
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
#pragma unroll 2
        for(uint row = numRows; row < numRowsAfterPadding; row++)
        {
            half* smem_row_ptr              = &smem_in[row * inputStride];
            ((half4*)smem_row_ptr)[lane_id] = zeros;
        }
    }

    __syncthreads();

    bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2>(smem_in,
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

enum : uint32_t
{
    // Data size parameters
    numRows   = 16,
    numCols   = 32,
    batchSize = 2,

    // Shared kernel template parameters
    K_WARP_SIZE      = AMDGCN_WAVE_SIZE,
    K_WARP_SIZE_LOG2 = Log2<K_WARP_SIZE>::value,
    K_TILE_DIM       = 16,
    K_TILE_DIM_LOG2  = Log2<K_TILE_DIM>::value,
    PAD              = 0,
    MEM_SKEW_SIZE    = 8,

    // Forward kernel template parameters
    M_BLOCKS              = numRows / K_TILE_DIM,
    K_BLOCKS              = numCols / K_TILE_DIM,
    SMEM_STRIDE           = K_BLOCKS * 16 + 8,
    SMEM_STRIDE_ACC       = M_BLOCKS * 16 + 8,
    WARPS_PER_THREADBLOCK = 128 / K_WARP_SIZE,
    THREADBLOCK_SIZE      = WARPS_PER_THREADBLOCK * K_WARP_SIZE,

    // Backward kernel template parameters
    TILE_SIZE            = K_TILE_DIM,
    K_WARPS_PER_BLOCK    = 128 / K_WARP_SIZE,
    K_NUM_THREADS        = K_WARPS_PER_BLOCK * K_WARP_SIZE,
    K_ROW_TILES_PER_STEP = 32 / K_TILE_DIM,
    K_COL_TILES_PER_STEP = 1
};

__host__ void dlrm_test(bool isBwd)
{
    // Allocate and initialize host matrices
    std::vector<float16_t> h_input, h_output, h_upstreamGrad, h_grad, h_bottomMlpGrad;

    const size_t tril_size = ((numCols * (numCols - 1)) / 2) + numRows;

    h_input.resize(numRows * numCols * batchSize);

    fill<float16_t>(h_input.data(), numRows, numCols, batchSize);

    if(!isBwd)
    {
        h_output.resize(tril_size * batchSize);
    }
    else
    {
        h_upstreamGrad.resize(tril_size * batchSize);
        h_grad.resize(numRows * numCols * batchSize);
        h_bottomMlpGrad.resize(numRows * batchSize);

        fill<float16_t>(h_upstreamGrad.data(), 1, tril_size, batchSize);
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
        uint numRowTiles = (numRows + K_TILE_DIM - 1) >> K_TILE_DIM_LOG2;
        uint numColTiles = (numCols + K_TILE_DIM - 1) >> K_TILE_DIM_LOG2;

        // number of rows and columns after padding
        uint numRowsAfterPadding = K_TILE_DIM << 1;
        uint numColsAfterPadding = numColTiles << K_TILE_DIM_LOG2;

        uint numRowSteps = numRowTiles / K_ROW_TILES_PER_STEP;
        uint numColSteps = numColTiles / K_COL_TILES_PER_STEP;

        // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
        const uint smemRowsPerWarp     = M_BLOCKS << 4;
        const uint smemElemsPerWarpMat = smemRowsPerWarp * SMEM_STRIDE;

        // output in FP32
        const uint smemElemsPerWarpAcc = M_BLOCKS * K_TILE_DIM * SMEM_STRIDE_ACC * 2;
        const uint smemElemsPerWarp    = (smemElemsPerWarpMat > smemElemsPerWarpAcc)
                                             ? smemElemsPerWarpMat
                                             : smemElemsPerWarpAcc;

        uint outputSize = numCols + (numRows * (numRows - 1) >> 1) + PAD;

        dlrmKernel = [d_input,
                      d_output,
                      numRowsAfterPadding,
                      numColsAfterPadding,
                      smemElemsPerWarp,
                      smemRowsPerWarp,
                      outputSize,
                      numRowSteps,
                      numColSteps]() {
            hipLaunchKernelGGL((dotBasedInteractFwdKernel<WARPS_PER_THREADBLOCK,
                                                          THREADBLOCK_SIZE,
                                                          M_BLOCKS,
                                                          K_BLOCKS,
                                                          SMEM_STRIDE,
                                                          SMEM_STRIDE_ACC,
                                                          K_WARP_SIZE,
                                                          K_WARP_SIZE_LOG2,
                                                          K_TILE_DIM,
                                                          K_TILE_DIM_LOG2>),
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
                               outputSize,
                               numRowSteps,
                               numColSteps,
                               PAD);
        };
    }
    else
    {
        // Backward kernel argument parameters
        const uint kWarpsPerBlockLog2 = Log2<K_WARPS_PER_BLOCK>::value;
        const uint tileSizeLog2       = Log2<TILE_SIZE>::value;

        uint inputDataBytes = sizeof(half);

        uint rowTilesPerStep = numRows > TILE_SIZE ? K_ROW_TILES_PER_STEP : 1;

        // num tiles
        uint numRowTiles = (numRows + TILE_SIZE - 1) >> tileSizeLog2;
        uint numColTiles = (numCols + TILE_SIZE - 1) >> tileSizeLog2;

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
        uint outputSizeElems = TILE_SIZE * TILE_SIZE * K_ROW_TILES_PER_STEP * K_COL_TILES_PER_STEP;
        uint outputSizeBytes = outputSizeElems * sizeof(float);

        // staging area size
        uint stagingAreaSizeBytes = outputSizeBytes > interactionUgrad2DSizeBytes
                                        ? outputSizeBytes
                                        : interactionUgrad2DSizeBytes;

        // Shared memory size
        uint wmmaSmemBytes            = numRowsAfterPadding * numRowsAfterPadding * inputDataBytes;
        uint sharedMemPerWarpSizeByte = inputSizeBytes + stagingAreaSizeBytes + wmmaSmemBytes;
        uint sharedMemSizeBytes       = K_WARPS_PER_BLOCK * sharedMemPerWarpSizeByte;

        uint numBlocks   = (batchSize + K_WARPS_PER_BLOCK - 1) >> kWarpsPerBlockLog2;
        uint numRowSteps = numRowTiles / rowTilesPerStep;
        uint numColSteps = numColTiles / K_COL_TILES_PER_STEP;

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
            hipLaunchKernelGGL((dotBasedInteractBwdKernel<K_WARPS_PER_BLOCK,
                                                          K_NUM_THREADS,
                                                          K_ROW_TILES_PER_STEP,
                                                          K_COL_TILES_PER_STEP,
                                                          K_WARP_SIZE,
                                                          K_WARP_SIZE_LOG2,
                                                          K_TILE_DIM,
                                                          K_TILE_DIM_LOG2>),
                               dim3(numBlocks),
                               dim3(K_NUM_THREADS),
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

        // for (int i = 0; i < h_input.size(); i++)
        //     std::cout << "Host: " << h_input[i] << '\n';

        for(int i = 0; i < outputRef.size(); i++)
            std::cout << "Host: " << outputRef[i] << ", Device: " << h_output[i] << '\n';

        compareEqual(h_output.data(), outputRef.data(), h_output.size());
    }
}

int main()
{
    dlrm_test(false);
    dlrm_test(true);
    return 0;
}
