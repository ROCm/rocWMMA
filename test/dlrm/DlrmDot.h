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

#ifndef DLRM_DOT_H
#define DLRM_DOT_H

#include "common.h"

template <typename DataT>
__device__ inline void store(DataT* dst, float* src)
{
    if(std::is_same<DataT, float16_t>::value)
        *dst = __float2half(*src);
    else
        *dst = *src;
}

template <typename DataT>
__device__ inline void store(DataT* dst, const float src)
{
    if(std::is_same<DataT, float16_t>::value)
        *dst = __float2half(src);
    else
        *dst = src;
}

template <uint TILE_DIM, uint M_BLOCKS, uint SMEM_STRIDE, uint SMEM_STRIDE_ACC, typename T>
__device__ inline void bmmTrilPadFwdKernel(T*   shmem,
                                           T*   gmem_output,
                                           uint num_rows,
                                           uint num_cols,
                                           uint smem_rows_per_warp,
                                           uint output_size,
                                           uint num_col_steps,
                                           uint pad,
                                           int  lane_id)
{
    wmma::fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[M_BLOCKS][M_BLOCKS];

    for(int i = 0; i < M_BLOCKS; i++)
    {
        for(int j = 0; j < M_BLOCKS; j++)
        {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for(int k_step = 0; k_step < num_col_steps; k_step++)
    {
        wmma::fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, T, row_major> a[M_BLOCKS];
        wmma::fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, T, col_major> b[M_BLOCKS];
        for(int j = 0; j < M_BLOCKS; j++)
        {
            int      base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
            const T* tile_ptr
                = reinterpret_cast<T*>(shmem) + (base_row * SMEM_STRIDE + k_step * 16);
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

    T*  gmem_interact_output = gmem_output + num_cols;
    int lastRowBlockOffset   = M_BLOCKS * 16 - smem_rows_per_warp;
    int srcLine              = 0;
    for(int i = 0; i < num_rows; ++i, ++srcLine)
    {
        if(i == ((M_BLOCKS - 1) * 16))
        {
            srcLine += lastRowBlockOffset;
        }
        if(lane_id < i)
        {
            uint offset = (i * (i - 1)) >> 1;
            store<T>(gmem_interact_output + offset + lane_id,
                     shmem_store + srcLine * SMEM_STRIDE_ACC + lane_id);
        }
    }

    // Padding
    if(lane_id < pad)
    {
        store<T>(gmem_output + lane_id + output_size - 1, 0.0f);
    }
}

template <typename T>
__device__ inline void trilBwdKernel(T*   smem_in,
                                     T*   smem_temp,
                                     T    zero,
                                     uint num_rows,
                                     uint num_rows_after_padding,
                                     uint interaction_ugrad_2D_stride,
                                     uint lane_id)
{
    if(lane_id < num_rows_after_padding)
    {
        uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
        uint ugrad_offset_1   = lane_id * interaction_ugrad_2D_stride;
        for(uint row = 0; row < num_rows; row++)
        {
            T ugrad_val = zero;
            if(row < lane_id && lane_id < num_rows)
            {
                ugrad_val                       = smem_in[ugrad_flat_index + row];
                smem_temp[ugrad_offset_1 + row] = ugrad_val;
            }
            if(row <= lane_id && lane_id < num_rows_after_padding)
            {
                smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
            }
        }
        for(uint row = num_rows; row < num_rows_after_padding; row++)
        {
            smem_temp[row * interaction_ugrad_2D_stride + lane_id] = zero;
        }
    }
}

template <uint TILE_DIM, uint ROW_TILES_PER_STEP, uint TILE_DIM_LOG_2, typename T>
__device__ inline void bmmBwdKernel(T*     smem_in,
                                    T*     smem_temp,
                                    float* smem_out,
                                    T*     gmem_grad,
                                    uint   num_rows,
                                    uint   num_cols,
                                    uint   num_col_steps,
                                    uint   input_stride,
                                    uint   interaction_ugrad_2D_stride,
                                    uint   lane_id)
{
    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, T, wmma::row_major>
        a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
    for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
    {
        for(uint j = 0; j < ROW_TILES_PER_STEP; j++)
        {
            const T* tile_ptr
                = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
            wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
        }
    }

    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[ROW_TILES_PER_STEP];
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, T, wmma::row_major>
        b[ROW_TILES_PER_STEP];
    for(int col_step = 0; col_step < num_col_steps; col_step++)
    {
        for(uint i = 0; i < ROW_TILES_PER_STEP; i++)
        {
            const T* tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
            wmma::fill_fragment(acc[i], 0.0f);
            wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
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

#ifdef __HIP_PLATFORM_HCC__
        __syncthreads();
#else
        __syncwarp();
#endif

        uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
        if(gmem_grad_col < num_cols)
        {
            for(uint i = 0; i < num_rows; i++)
            {
                store<T>(&gmem_grad[i * num_cols + gmem_grad_col],
                         &smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
            }
        }
    }
}

template <typename DataT,
          uint WARPS_PER_BLOCK,
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
    void dotBasedInteractFwdKernelNonAligned(const DataT* __restrict input,
                                             DataT* __restrict output,
                                             uint batch_size,
                                             uint num_rows,
                                             uint num_cols,
                                             uint num_rows_after_padding,
                                             uint num_cols_after_padding,
                                             uint smem_elems_per_warp,
                                             uint smem_rows_per_warp,
                                             uint output_size,
                                             uint num_row_steps,
                                             uint num_col_steps,
                                             uint pad)
{
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    int  sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    int lane_id = threadIdx.x & (WARP_SIZE - 1);

    HIP_DYNAMIC_SHARED(void*, shmem_dynamic)
    DataT* shmem = reinterpret_cast<DataT*>(shmem_dynamic) + (warp_id * smem_elems_per_warp);

    const DataT* sample_input = input + num_rows * num_cols * sample_id;
    for(uint i = 0; i < num_rows; ++i, sample_input += num_cols)
    {
        for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
        {
            (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
        }
    }

    uint idx = lane_id + num_cols;
    if(std::is_same<DataT, float16_t>::value)
    {
        if(idx < num_cols_after_padding)
        {
            for(int i = 0; i < num_rows; ++i)
            {
                (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
            }
        }
        half4 zeros;
        zeros.vals[0] = __float2half2_rn(0);
        zeros.vals[1] = __float2half2_rn(0);

        if(lane_id < (num_cols_after_padding >> 2))
        {
            for(int i = num_rows; i < num_rows_after_padding; i++)
            {
                ((half4*)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
            }
        }
    }
    else
    {
        if(idx < num_cols_after_padding)
        {
            for(int i = 0; i < num_rows; ++i)
            {
                (shmem + i * SMEM_STRIDE)[idx] = 0;
            }
        }

        for(int i = num_rows; i < num_rows_after_padding; i++)
        {
            for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
            {
                (shmem + i * SMEM_STRIDE)[idx] = 0;
            }
        }
    }

    __syncthreads();

    DataT* gmem_output = output + output_size * sample_id;
    for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
    {
        gmem_output[idx] = shmem[idx];
    }

    bmmTrilPadFwdKernel<TILE_DIM, M_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC, DataT>(shmem,
                                                                                 gmem_output,
                                                                                 num_rows,
                                                                                 num_cols,
                                                                                 smem_rows_per_warp,
                                                                                 output_size,
                                                                                 num_col_steps,
                                                                                 pad,
                                                                                 lane_id);
}

template <typename DataT,
          uint WARPS_PER_BLOCK,
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
    void dotBasedInteractFwdKernel(const DataT* __restrict input,
                                   DataT* __restrict output,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols,
                                   uint num_rows_after_padding,
                                   uint num_cols_after_padding,
                                   uint smem_elems_per_warp,
                                   uint smem_rows_per_warp,
                                   uint output_size,
                                   uint num_row_steps,
                                   uint num_col_steps,
                                   uint pad)
{
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    int  sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    int lane_id = threadIdx.x & (WARP_SIZE - 1);

    HIP_DYNAMIC_SHARED(void*, shmem_dynamic)
    // reinterpret cast to dataT
    DataT* shmem = reinterpret_cast<DataT*>(shmem_dynamic) + (warp_id * smem_elems_per_warp);

    const DataT* sample_input = input + num_rows * num_cols * sample_id;
    if(lane_id < (num_cols >> 1))
    {
        for(int i = 0; i < num_rows; ++i, sample_input += num_cols)
        {
            ((float2*)(shmem + i * SMEM_STRIDE))[lane_id] = ((float2*)sample_input)[lane_id];
        }
    }

    uint idx = lane_id + num_cols;
    if(std::is_same<DataT, float16_t>::value)
    {
        if(idx < num_cols_after_padding)
        {
            for(int i = 0; i < num_rows; ++i)
            {
                (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
            }
        }
        half4 zeros;
        zeros.vals[0] = __float2half2_rn(0);
        zeros.vals[1] = __float2half2_rn(0);

        if(lane_id < (num_cols_after_padding >> 2))
        {
            for(int i = num_rows; i < num_rows_after_padding; i++)
            {
                ((half4*)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
            }
        }
    }
    else
    {
        if(idx < num_cols_after_padding)
        {
            for(int i = 0; i < num_rows; ++i)
            {
                (shmem + i * SMEM_STRIDE)[idx] = 0;
            }
        }

        for(int i = num_rows; i < num_rows_after_padding; i++)
        {
            for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
            {
                (shmem + i * SMEM_STRIDE)[idx] = 0;
            }
        }
    }

    __syncthreads();

    DataT* gmem_output = output + output_size * sample_id;
    if(lane_id < (num_cols >> 1))
    {
        ((float2*)gmem_output)[lane_id] = ((float2*)shmem)[lane_id];
    }

    bmmTrilPadFwdKernel<TILE_DIM, M_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC, DataT>(shmem,
                                                                                 gmem_output,
                                                                                 num_rows,
                                                                                 num_cols,
                                                                                 smem_rows_per_warp,
                                                                                 output_size,
                                                                                 num_col_steps,
                                                                                 pad,
                                                                                 lane_id);
}

template <typename DataT,
          uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernelNonAligned(const DataT* __restrict input,
                                             const DataT* __restrict upstream_grad,
                                             DataT* __restrict grad,
                                             DataT* __restrict bottom_mlp_grad,
                                             uint batch_size,
                                             uint num_rows,
                                             uint num_cols,
                                             uint num_rows_after_padding,
                                             uint num_cols_after_padding,
                                             uint sample_size,
                                             uint interaction_ugrad_size,
                                             uint interaction_ugrad_size_with_padding,
                                             uint interaction_ugrad_2D_size_elems,
                                             uint interaction_ugrad_2D_stride,
                                             uint input_size_elems,
                                             uint input_stride,
                                             uint num_row_steps,
                                             uint num_col_steps,
                                             uint row_tiles_per_step,
                                             uint shared_mem_per_warp_size_byte)
{
    HIP_DYNAMIC_SHARED(void*, shared_mem)
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    uint lane_id = threadIdx.x & (WARP_SIZE - 1);
    // ">> 2" to convert to DataT pointer
    uint smem_warp_offset
        = warp_id
          * (shared_mem_per_warp_size_byte >> (std::is_same<DataT, float16_t>::value ? 1 : 2));
    uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

    DataT* smem_in   = reinterpret_cast<DataT*>(shared_mem) + smem_warp_offset + wmma_offset;
    DataT* smem_temp = smem_in + input_size_elems;
    float* smem_out  = reinterpret_cast<float*>(smem_temp);

    // Global memory pointers for the current sample
    // Input
    uint         gmem_input_sample_offset = sample_id * sample_size;
    const DataT* gmem_input               = &input[gmem_input_sample_offset];

    // Interaction Gradient
    const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
    DataT*      gmem_grad               = &grad[gmem_grad_sample_offset];

    // Bottom MLP gradient
    DataT* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

    // Upstream gradient vector
    uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
    const DataT* gmem_ugrad       = &upstream_grad[gmem_ugrad_sample_offset];

    // Upstream gradient vector for interactions
    const DataT* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

    // upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
    for(uint idx = lane_id; idx < interaction_ugrad_size; idx += WARP_SIZE)
    {
        smem_in[idx] = gmem_ugrad_interactions[idx];
    }

    __syncthreads();

    // Form the 2D ugrad matrix.
    trilBwdKernel(smem_in,
                  smem_temp,
                  static_cast<DataT>(0.0),
                  num_rows,
                  num_rows_after_padding,
                  interaction_ugrad_2D_stride,
                  lane_id);

    __syncthreads();

    // Input -> Shared Memory

    for(uint row = 0; row < num_rows; row++)
    {
        DataT*       smem_row_ptr = &smem_in[row * input_stride];
        const DataT* gmem_row_ptr = &gmem_input[row * num_cols];
        for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
        {
            smem_row_ptr[idx] = gmem_row_ptr[idx];
        }
        uint idx = lane_id + num_cols;
        if(idx < num_cols_after_padding)
        {
            if(std::is_same<DataT, float16_t>::value)
                smem_row_ptr[idx] = __float2half(0);
            else
                smem_row_ptr[idx] = 0;
        }
    }

#pragma unroll 2
    for(uint row = num_rows; row < num_rows_after_padding; row++)
    {
        DataT* smem_row_ptr = &smem_in[row * input_stride];
        for(uint idx = lane_id; idx < num_cols_after_padding; idx += WARP_SIZE)
        {
            if(std::is_same<DataT, float16_t>::value)
                smem_row_ptr[idx] = __float2half(0);
            else
                smem_row_ptr[idx] = 0;
        }
    }

    __syncthreads();

    bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, DataT>(smem_in,
                                                                      smem_temp,
                                                                      smem_out,
                                                                      gmem_grad,
                                                                      num_rows,
                                                                      num_cols,
                                                                      num_col_steps,
                                                                      input_stride,
                                                                      interaction_ugrad_2D_stride,
                                                                      lane_id);

    for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
    {
        gmem_mlp_grad[idx] = gmem_ugrad[idx];
    }
}

template <typename DataT,
          uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernel(const DataT* __restrict input,
                                   const DataT* __restrict upstream_grad,
                                   DataT* __restrict grad,
                                   DataT* __restrict bottom_mlp_grad,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols,
                                   uint num_rows_after_padding,
                                   uint num_cols_after_padding,
                                   uint sample_size,
                                   uint interaction_ugrad_size,
                                   uint interaction_ugrad_size_with_padding,
                                   uint interaction_ugrad_2D_size_elems,
                                   uint interaction_ugrad_2D_stride,
                                   uint input_size_elems,
                                   uint input_stride,
                                   uint num_row_steps,
                                   uint num_col_steps,
                                   uint row_tiles_per_step,
                                   uint shared_mem_per_warp_size_byte)
{
    HIP_DYNAMIC_SHARED(void*, shared_mem)
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    uint lane_id = threadIdx.x & (WARP_SIZE - 1);
    // ">> 2" to convert to DataT pointer
    uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 2);
    uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

    DataT* smem_in   = reinterpret_cast<DataT*>(shared_mem) + smem_warp_offset + wmma_offset;
    DataT* smem_temp = smem_in + input_size_elems;
    float* smem_out  = reinterpret_cast<float*>(smem_temp);

    // Global memory pointers for the current sample
    // Input
    uint         gmem_input_sample_offset = sample_id * sample_size;
    const DataT* gmem_input               = &input[gmem_input_sample_offset];

    // Interaction Gradient
    const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
    DataT*      gmem_grad               = &grad[gmem_grad_sample_offset];

    // Bottom MLP gradient
    DataT* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

    // Upstream gradient vector
    uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
    const DataT* gmem_ugrad       = &upstream_grad[gmem_ugrad_sample_offset];

    // Upstream gradient vector for interactions
    const DataT* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

    // upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
    for(uint idx = lane_id; idx < (interaction_ugrad_size >> 2); idx += WARP_SIZE)
    {
        ((float4*)smem_in)[idx] = ((float4*)gmem_ugrad_interactions)[idx];
    }
    uint offset = (interaction_ugrad_size >> 2) << 2;
    for(uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += WARP_SIZE)
    {
        smem_in[idx] = gmem_ugrad_interactions[idx];
    }

    __syncthreads();

    // Form the 2D ugrad matrix.
    trilBwdKernel(smem_in,
                  smem_temp,
                  static_cast<DataT>(0.0),
                  num_rows,
                  num_rows_after_padding,
                  interaction_ugrad_2D_stride,
                  lane_id);

    __syncthreads();

    // Input -> Shared Memory
    if(lane_id < (num_cols >> 1))
    {
        for(uint row = 0; row < num_rows; row++)
        {
            DataT*       smem_row_ptr        = &smem_in[row * input_stride];
            const DataT* gmem_row_ptr        = &gmem_input[row * num_cols];
            ((float2*)smem_row_ptr)[lane_id] = ((float2*)gmem_row_ptr)[lane_id];
        }
    }

    uint idx = lane_id + num_cols;
    if(idx < num_cols_after_padding)
    {
        for(uint row = 0; row < num_rows; row++)
        {
            DataT* smem_row_ptr = &smem_in[row * input_stride];
            smem_row_ptr[idx]   = 0;
        }
    }

    if(std::is_same<DataT, float16_t>::value)
    {
        half4 zeros;
        zeros.vals[0] = __float2half2_rn(0);
        zeros.vals[1] = __float2half2_rn(0);

        if(lane_id < (num_cols_after_padding >> 2))
        {
#pragma unroll 2
            for(uint row = num_rows; row < num_rows_after_padding; row++)
            {
                DataT* smem_row_ptr             = &smem_in[row * input_stride];
                ((half4*)smem_row_ptr)[lane_id] = zeros;
            }
        }
    }
    else
    {
        float4 zeros;
        zeros.data[0] = 0;
        zeros.data[1] = 0;
        zeros.data[2] = 0;
        zeros.data[3] = 0;

        if(lane_id < (num_cols_after_padding >> 2))
        {
#pragma unroll 2
            for(uint row = num_rows; row < num_rows_after_padding; row++)
            {
                DataT* smem_row_ptr              = &smem_in[row * input_stride];
                ((float4*)smem_row_ptr)[lane_id] = zeros;
            }
        }
    }

    __syncthreads();

    bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, DataT>(smem_in,
                                                                      smem_temp,
                                                                      smem_out,
                                                                      gmem_grad,
                                                                      num_rows,
                                                                      num_cols,
                                                                      num_col_steps,
                                                                      input_stride,
                                                                      interaction_ugrad_2D_stride,
                                                                      lane_id);

    if(lane_id < (num_cols >> 1))
    {
        ((float2*)gmem_mlp_grad)[lane_id] = ((float2*)gmem_ugrad)[lane_id];
    }
}

#endif // DLRM_DOT_H
