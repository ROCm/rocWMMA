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

#ifndef DLRM_DOT_FWD_H
#define DLRM_DOT_FWD_H

#include "./Common.h"

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

#endif // DLRM_DOT_FWD_H
