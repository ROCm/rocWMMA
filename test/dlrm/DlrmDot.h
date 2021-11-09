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
        *dst = *src
}

template <typename DataT>
__device__ inline void store(DataT* dst, const float src)
{
    if(std::is_same<DataT, float16_t>::value)
        *dst = __float2half(*src);
    else
        *dst = *src
}

// __device__ inline void store(float* dst, float* src)
// {
//     *dst = *src;
// }

// __device__ inline void store(float* dst, const float src)
// {
//     *dst = src;
// }

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
            store(gmem_interact_output + offset + lane_id,
                  shmem_store + srcLine * SMEM_STRIDE_ACC + lane_id);
        }
    }

    // Padding
    if(lane_id < pad)
    {
        store(gmem_output + lane_id + output_size - 1, 0.0f);
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
                store(&gmem_grad[i * num_cols + gmem_grad_col],
                      &smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
            }
        }
    }
}

// template <uint WARPS_PER_BLOCK,
//           uint THREADBLOCK_SIZE,
//           uint M_BLOCKS,
//           uint K_BLOCKS,
//           uint SMEM_STRIDE,
//           uint SMEM_STRIDE_ACC,
//           uint WARP_SIZE,
//           uint WARP_SIZE_LOG_2,
//           uint TILE_DIM,
//           uint TILE_DIM_LOG_2>
// __launch_bounds__(THREADBLOCK_SIZE) __global__
//     void dotBasedInteractFwdKernelNonAligned(const __half* __restrict input,
//                                              __half* __restrict output,
//                                              uint batch_size,
//                                              uint num_rows,
//                                              uint num_cols,
//                                              uint num_rows_after_padding,
//                                              uint num_cols_after_padding,
//                                              uint smem_elems_per_warp,
//                                              uint smem_rows_per_warp,
//                                              uint output_size,
//                                              uint num_row_steps,
//                                              uint num_col_steps,
//                                              uint pad)
// {
//     uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
//     int  sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
//     if(sample_id >= batch_size)
//     {
//         return;
//     }
//     int lane_id = threadIdx.x & (WARP_SIZE - 1);

//     HIP_DYNAMIC_SHARED(half, shmem_dynamic_half)
//     half* shmem = shmem_dynamic_half + (warp_id * smem_elems_per_warp);

//     const half* sample_input = input + num_rows * num_cols * sample_id;
//     for(uint i = 0; i < num_rows; ++i, sample_input += num_cols)
//     {
//         for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
//         {
//             (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
//         }
//     }

//     uint idx = lane_id + num_cols;
//     if(idx < num_cols_after_padding)
//     {
//         for(int i = 0; i < num_rows; ++i)
//         {
//             (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
//         }
//     }

//     half4 zeros;
// #ifdef __HIP_PLATFORM_HCC__
//     zeros.vals[0] = __float2half2_rn(0);
//     zeros.vals[1] = __float2half2_rn(0);
// #else
//     half4 zeros;
//     zeros.vals[0].x = __float2half(0);
//     zeros.vals[0].y = __float2half(0);
//     zeros.vals[1].x = __float2half(0);
//     zeros.vals[1].y = __float2half(0);
// #endif
//     if(lane_id < (num_cols_after_padding >> 2))
//     {
//         for(int i = num_rows; i < num_rows_after_padding; i++)
//         {
//             ((half4*)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
//         }
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     half* gmem_output = output + output_size * sample_id;
//     for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
//     {
//         gmem_output[idx] = shmem[idx];
//     }

//     bmmTrilPadFwdKernel<TILE_DIM, M_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC, __half>(
//         shmem,
//         gmem_output,
//         num_rows,
//         num_cols,
//         smem_rows_per_warp,
//         output_size,
//         num_col_steps,
//         pad,
//         lane_id);
// }

// template <uint WARPS_PER_BLOCK,
//           uint THREADBLOCK_SIZE,
//           uint M_BLOCKS,
//           uint K_BLOCKS,
//           uint SMEM_STRIDE,
//           uint SMEM_STRIDE_ACC,
//           uint WARP_SIZE,
//           uint WARP_SIZE_LOG_2,
//           uint TILE_DIM,
//           uint TILE_DIM_LOG_2>
// __launch_bounds__(THREADBLOCK_SIZE) __global__
//     void dotBasedInteractFwdKernel(const __half* __restrict input,
//                                    __half* __restrict output,
//                                    uint batch_size,
//                                    uint num_rows,
//                                    uint num_cols,
//                                    uint num_rows_after_padding,
//                                    uint num_cols_after_padding,
//                                    uint smem_elems_per_warp,
//                                    uint smem_rows_per_warp,
//                                    uint output_size,
//                                    uint num_row_steps,
//                                    uint num_col_steps,
//                                    uint pad)
// {
//     uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
//     int  sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
//     if(sample_id >= batch_size)
//     {
//         return;
//     }
//     int lane_id = threadIdx.x & (WARP_SIZE - 1);

//     HIP_DYNAMIC_SHARED(half, shmem_dynamic_half)
//     half* shmem = shmem_dynamic_half + (warp_id * smem_elems_per_warp);

//     const half* sample_input = input + num_rows * num_cols * sample_id;
//     if(lane_id < (num_cols >> 2))
//     {
//         for(int i = 0; i < num_rows; ++i, sample_input += num_cols)
//         {
//             ((float2*)(shmem + i * SMEM_STRIDE))[lane_id] = ((float2*)sample_input)[lane_id];
//         }
//     }

//     uint idx = lane_id + num_cols;
//     if(idx < num_cols_after_padding)
//     {
//         for(int i = 0; i < num_rows; ++i)
//         {
//             (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
//         }
//     }

//     half4 zeros;
// #ifdef __HIP_PLATFORM_HCC__
//     zeros.vals[0] = __float2half2_rn(0);
//     zeros.vals[1] = __float2half2_rn(0);
// #else
//     zeros.vals[0].x = __float2half(0);
//     zeros.vals[0].y = __float2half(0);
//     zeros.vals[1].x = __float2half(0);
//     zeros.vals[1].y = __float2half(0);
// #endif
//     if(lane_id < (num_cols_after_padding >> 2))
//     {
//         for(int i = num_rows; i < num_rows_after_padding; i++)
//         {
//             ((half4*)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
//         }
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     half* gmem_output = output + output_size * sample_id;
//     if(lane_id < (num_cols >> 2))
//     {
//         ((float2*)gmem_output)[lane_id] = ((float2*)shmem)[lane_id];
//     }

//     bmmTrilPadFwdKernel<TILE_DIM, M_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC, __half>(
//         shmem,
//         gmem_output,
//         num_rows,
//         num_cols,
//         smem_rows_per_warp,
//         output_size,
//         num_col_steps,
//         pad,
//         lane_id);
// }

#ifdef __HIP_PLATFORM_HCC__
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
#endif // __HIP_PLATFORM_HCC__

// template <uint WARPS_PER_BLOCK,
//           uint THREADBLOCK_SIZE,
//           uint ROW_TILES_PER_STEP,
//           uint COL_TILES_PER_STEP,
//           uint WARP_SIZE,
//           uint WARP_SIZE_LOG_2,
//           uint TILE_DIM,
//           uint TILE_DIM_LOG_2>
// __launch_bounds__(THREADBLOCK_SIZE) __global__
//     void dotBasedInteractBwdKernelNonAligned(const __half* __restrict input,
//                                              const __half* __restrict upstream_grad,
//                                              half* __restrict grad,
//                                              half* __restrict bottom_mlp_grad,
//                                              uint batch_size,
//                                              uint num_rows,
//                                              uint num_cols,
//                                              uint num_rows_after_padding,
//                                              uint num_cols_after_padding,
//                                              uint sample_size,
//                                              uint interaction_ugrad_size,
//                                              uint interaction_ugrad_size_with_padding,
//                                              uint interaction_ugrad_2D_size_elems,
//                                              uint interaction_ugrad_2D_stride,
//                                              uint input_size_elems,
//                                              uint input_stride,
//                                              uint num_row_steps,
//                                              uint num_col_steps,
//                                              uint row_tiles_per_step,
//                                              uint shared_mem_per_warp_size_byte)
// {
//     HIP_DYNAMIC_SHARED(half, shared_mem_half)
//     uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
//     uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
//     if(sample_id >= batch_size)
//     {
//         return;
//     }
//     uint lane_id = threadIdx.x & (WARP_SIZE - 1);
//     // ">> 1" to convert to half pointer
//     uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);
//     uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

//     half*  smem_in   = shared_mem_half + smem_warp_offset + wmma_offset;
//     half*  smem_temp = smem_in + input_size_elems;
//     float* smem_out  = reinterpret_cast<float*>(smem_temp);

//     // Global memory pointers for the current sample
//     // Input
//     uint        gmem_input_sample_offset = sample_id * sample_size;
//     const half* gmem_input               = &input[gmem_input_sample_offset];

//     // Interaction Gradient
//     const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
//     half*       gmem_grad               = &grad[gmem_grad_sample_offset];

//     // Bottom MLP gradient
//     half* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

//     // Upstream gradient vector
//     uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
//     const half* gmem_ugrad        = &upstream_grad[gmem_ugrad_sample_offset];

//     // Upstream gradient vector for interactions
//     const half* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

//     // upstream grad -> shared memory (place in input section temporarily)
// #pragma unroll
//     for(uint idx = lane_id; idx < interaction_ugrad_size; idx += WARP_SIZE)
//     {
//         smem_in[idx] = gmem_ugrad_interactions[idx];
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     // Form the 2D ugrad matrix.
//     trilBwdKernel(smem_in,
//                   smem_temp,
//                   __float2half(0),
//                   num_rows,
//                   num_rows_after_padding,
//                   interaction_ugrad_2D_stride,
//                   lane_id);

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     // Input -> Shared Memory

//     for(uint row = 0; row < num_rows; row++)
//     {
//         half*       smem_row_ptr = &smem_in[row * input_stride];
//         const half* gmem_row_ptr = &gmem_input[row * num_cols];
//         for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
//         {
//             smem_row_ptr[idx] = gmem_row_ptr[idx];
//         }
//         uint idx = lane_id + num_cols;
//         if(idx < num_cols_after_padding)
//         {
//             smem_row_ptr[idx] = __float2half(0);
//         }
//     }

// #pragma unroll 2
//     for(uint row = num_rows; row < num_rows_after_padding; row++)
//     {
//         half* smem_row_ptr = &smem_in[row * input_stride];
//         for(uint idx = lane_id; idx < num_cols_after_padding; idx += WARP_SIZE)
//         {
//             smem_row_ptr[idx] = __float2half(0);
//         }
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, __half>(smem_in,
//                                                                        smem_temp,
//                                                                        smem_out,
//                                                                        gmem_grad,
//                                                                        num_rows,
//                                                                        num_cols,
//                                                                        num_col_steps,
//                                                                        input_stride,
//                                                                        interaction_ugrad_2D_stride,
//                                                                        lane_id);

//     for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
//     {
//         gmem_mlp_grad[idx] = gmem_ugrad[idx];
//     }
// }

// template <uint WARPS_PER_BLOCK,
//           uint THREADBLOCK_SIZE,
//           uint ROW_TILES_PER_STEP,
//           uint COL_TILES_PER_STEP,
//           uint WARP_SIZE,
//           uint WARP_SIZE_LOG_2,
//           uint TILE_DIM,
//           uint TILE_DIM_LOG_2>
// __launch_bounds__(THREADBLOCK_SIZE) __global__
//     void dotBasedInteractBwdKernel(const __half* __restrict input,
//                                    const __half* __restrict upstream_grad,
//                                    half* __restrict grad,
//                                    half* __restrict bottom_mlp_grad,
//                                    uint batch_size,
//                                    uint num_rows,
//                                    uint num_cols,
//                                    uint num_rows_after_padding,
//                                    uint num_cols_after_padding,
//                                    uint sample_size,
//                                    uint interaction_ugrad_size,
//                                    uint interaction_ugrad_size_with_padding,
//                                    uint interaction_ugrad_2D_size_elems,
//                                    uint interaction_ugrad_2D_stride,
//                                    uint input_size_elems,
//                                    uint input_stride,
//                                    uint num_row_steps,
//                                    uint num_col_steps,
//                                    uint row_tiles_per_step,
//                                    uint shared_mem_per_warp_size_byte)
// {
//     HIP_DYNAMIC_SHARED(half, shared_mem_half)
//     uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
//     uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
//     if(sample_id >= batch_size)
//     {
//         return;
//     }
//     uint lane_id = threadIdx.x & (WARP_SIZE - 1);
//     // ">> 1" to convert to half pointer
//     uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);
//     uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

//     half*  smem_in   = shared_mem_half + smem_warp_offset + wmma_offset;
//     half*  smem_temp = smem_in + input_size_elems;
//     float* smem_out  = reinterpret_cast<float*>(smem_temp);

//     // Global memory pointers for the current sample
//     // Input
//     uint        gmem_input_sample_offset = sample_id * sample_size;
//     const half* gmem_input               = &input[gmem_input_sample_offset];

//     // Interaction Gradient
//     const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
//     half*       gmem_grad               = &grad[gmem_grad_sample_offset];

//     // Bottom MLP gradient
//     half* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

//     // Upstream gradient vector
//     uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
//     const half* gmem_ugrad        = &upstream_grad[gmem_ugrad_sample_offset];

//     // Upstream gradient vector for interactions
//     const half* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

//     // upstream grad -> shared memory (place in input section temporarily)
// #pragma unroll
//     for(uint idx = lane_id; idx < (interaction_ugrad_size >> 3); idx += WARP_SIZE)
//     {
//         ((float4*)smem_in)[idx] = ((float4*)gmem_ugrad_interactions)[idx];
//     }
//     uint offset = (interaction_ugrad_size >> 3) << 3;
//     for(uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += WARP_SIZE)
//     {
//         smem_in[idx] = gmem_ugrad_interactions[idx];
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     // Form the 2D ugrad matrix.
//     trilBwdKernel(smem_in,
//                   smem_temp,
//                   __float2half(0),
//                   num_rows,
//                   num_rows_after_padding,
//                   interaction_ugrad_2D_stride,
//                   lane_id);

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     // Input -> Shared Memory

//     if(lane_id < (num_cols >> 2))
//     {
//         for(uint row = 0; row < num_rows; row++)
//         {
//             half*       smem_row_ptr         = &smem_in[row * input_stride];
//             const half* gmem_row_ptr         = &gmem_input[row * num_cols];
//             ((float2*)smem_row_ptr)[lane_id] = ((float2*)gmem_row_ptr)[lane_id];
//         }
//     }

//     uint idx = lane_id + num_cols;
//     if(idx < num_cols_after_padding)
//     {
//         for(uint row = 0; row < num_rows; row++)
//         {
//             half* smem_row_ptr = &smem_in[row * input_stride];
//             smem_row_ptr[idx]  = __float2half(0);
//         }
//     }

//     half4 zeros;
// #ifdef __HIP_PLATFORM_HCC__
//     zeros.vals[0] = __float2half2_rn(0);
//     zeros.vals[1] = __float2half2_rn(0);
// #else
//     half4 zeros;
//     zeros.vals[0].x = __float2half(0);
//     zeros.vals[0].y = __float2half(0);
//     zeros.vals[1].x = __float2half(0);
//     zeros.vals[1].y = __float2half(0);
// #endif

//     if(lane_id < (num_cols_after_padding >> 2))
//     {
// #pragma unroll 2
//         for(uint row = num_rows; row < num_rows_after_padding; row++)
//         {
//             half* smem_row_ptr              = &smem_in[row * input_stride];
//             ((half4*)smem_row_ptr)[lane_id] = zeros;
//         }
//     }

// #ifdef __HIP_PLATFORM_HCC__
//     __syncthreads();
// #else
//     __syncwarp();
// #endif

//     bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, __half>(smem_in,
//                                                                        smem_temp,
//                                                                        smem_out,
//                                                                        gmem_grad,
//                                                                        num_rows,
//                                                                        num_cols,
//                                                                        num_col_steps,
//                                                                        input_stride,
//                                                                        interaction_ugrad_2D_stride,
//                                                                        lane_id);

//     if(lane_id < (num_cols >> 2))
//     {
//         ((float2*)gmem_mlp_grad)[lane_id] = ((float2*)gmem_ugrad)[lane_id];
//     }
// }

#ifdef __HIP_PLATFORM_HCC__
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
    void dotBasedInteractBwdKernelNonAligned(const float* __restrict input,
                                             const float* __restrict upstream_grad,
                                             float* __restrict grad,
                                             float* __restrict bottom_mlp_grad,
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
    HIP_DYNAMIC_SHARED(float, shared_mem)
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    uint lane_id = threadIdx.x & (WARP_SIZE - 1);
    // ">> 2" to convert to float pointer
    uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 2);
    uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

    float* smem_in   = shared_mem + smem_warp_offset + wmma_offset;
    float* smem_temp = smem_in + input_size_elems;
    float* smem_out  = smem_temp;

    // Global memory pointers for the current sample
    // Input
    uint         gmem_input_sample_offset = sample_id * sample_size;
    const float* gmem_input               = &input[gmem_input_sample_offset];

    // Interaction Gradient
    const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
    float*      gmem_grad               = &grad[gmem_grad_sample_offset];

    // Bottom MLP gradient
    float* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

    // Upstream gradient vector
    uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
    const float* gmem_ugrad       = &upstream_grad[gmem_ugrad_sample_offset];

    // Upstream gradient vector for interactions
    const float* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

    // upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
    for(uint idx = lane_id; idx < interaction_ugrad_size; idx += WARP_SIZE)
    {
        smem_in[idx] = gmem_ugrad_interactions[idx];
    }

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    // Form the 2D ugrad matrix.
    trilBwdKernel(smem_in,
                  smem_temp,
                  0.0f,
                  num_rows,
                  num_rows_after_padding,
                  interaction_ugrad_2D_stride,
                  lane_id);

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    // Input -> Shared Memory

    for(uint row = 0; row < num_rows; row++)
    {
        float*       smem_row_ptr = &smem_in[row * input_stride];
        const float* gmem_row_ptr = &gmem_input[row * num_cols];
        for(uint idx = lane_id; idx < num_cols; idx += WARP_SIZE)
        {
            smem_row_ptr[idx] = gmem_row_ptr[idx];
        }
        uint idx = lane_id + num_cols;
        if(idx < num_cols_after_padding)
        {
            smem_row_ptr[idx] = 0;
        }
    }

#pragma unroll 2
    for(uint row = num_rows; row < num_rows_after_padding; row++)
    {
        float* smem_row_ptr = &smem_in[row * input_stride];
        for(uint idx = lane_id; idx < num_cols_after_padding; idx += WARP_SIZE)
        {
            smem_row_ptr[idx] = 0;
        }
    }

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, float>(smem_in,
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
    void dotBasedInteractBwdKernel(const float* __restrict input,
                                   const float* __restrict upstream_grad,
                                   float* __restrict grad,
                                   float* __restrict bottom_mlp_grad,
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
    HIP_DYNAMIC_SHARED(float, shared_mem)
    uint warp_id   = (threadIdx.x >> WARP_SIZE_LOG_2);
    uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(sample_id >= batch_size)
    {
        return;
    }
    uint lane_id = threadIdx.x & (WARP_SIZE - 1);
    // ">> 2" to convert to float pointer
    uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 2);
    uint wmma_offset = num_rows_after_padding * num_rows_after_padding; // matrix_a is row_major

    float* smem_in   = shared_mem + smem_warp_offset + wmma_offset;
    float* smem_temp = smem_in + input_size_elems;
    float* smem_out  = smem_temp;

    // Global memory pointers for the current sample
    // Input
    uint         gmem_input_sample_offset = sample_id * sample_size;
    const float* gmem_input               = &input[gmem_input_sample_offset];

    // Interaction Gradient
    const uint& gmem_grad_sample_offset = gmem_input_sample_offset;
    float*      gmem_grad               = &grad[gmem_grad_sample_offset];

    // Bottom MLP gradient
    float* gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

    // Upstream gradient vector
    uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
    const float* gmem_ugrad       = &upstream_grad[gmem_ugrad_sample_offset];

    // Upstream gradient vector for interactions
    const float* gmem_ugrad_interactions = &gmem_ugrad[num_cols];

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

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    // Form the 2D ugrad matrix.
    trilBwdKernel(smem_in,
                  smem_temp,
                  0.0f,
                  num_rows,
                  num_rows_after_padding,
                  interaction_ugrad_2D_stride,
                  lane_id);

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    // Input -> Shared Memory
    if(lane_id < (num_cols >> 1))
    {
        for(uint row = 0; row < num_rows; row++)
        {
            float*       smem_row_ptr        = &smem_in[row * input_stride];
            const float* gmem_row_ptr        = &gmem_input[row * num_cols];
            ((float2*)smem_row_ptr)[lane_id] = ((float2*)gmem_row_ptr)[lane_id];
        }
    }

    uint idx = lane_id + num_cols;
    if(idx < num_cols_after_padding)
    {
        for(uint row = 0; row < num_rows; row++)
        {
            float* smem_row_ptr = &smem_in[row * input_stride];
            smem_row_ptr[idx]   = 0;
        }
    }

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
            float* smem_row_ptr              = &smem_in[row * input_stride];
            ((float4*)smem_row_ptr)[lane_id] = zeros;
        }
    }

#ifdef __HIP_PLATFORM_HCC__
    __syncthreads();
#else
    __syncwarp();
#endif

    bmmBwdKernel<TILE_DIM, ROW_TILES_PER_STEP, TILE_DIM_LOG_2, float>(smem_in,
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
#endif // __HIP_PLATFORM_HCC__

inline void dotBasedInteractFwd(const void* input,
                                const void* bottom_mlp_output,
                                void*       output,
                                uint        batch_size,
                                uint        num_rows,
                                uint        num_cols,
                                uint        pad,
                                bool        amp_train)
{
#ifdef __HIP_PLATFORM_HCC__
    const uint kWarpSize = 64;
#else
    const uint kWarpSize = 32;
#endif
    const uint kWarpSizeLog2         = Log2<kWarpSize>::value;
    const uint kTileDim              = 16;
    const uint kTileDimLog2          = Log2<kTileDim>::value;
    const uint warps_per_threadblock = 128 / kWarpSize;
    const uint threadblock_size      = warps_per_threadblock * kWarpSize;
    const uint kRowTilesPerStep      = 2;
    const uint kColTilesPerStep      = 1;

    // num tiles
    uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
    uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

    // number of rows and columns after padding
    uint num_rows_after_padding = kTileDim << 1;
    uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

    uint num_row_steps = num_row_tiles / kRowTilesPerStep;
    uint num_col_steps = num_col_tiles / kColTilesPerStep;

    const uint K_BLOCKS    = 8;
    const uint M_BLOCKS    = 2;
    const uint SKEW_HALF   = ((K_BLOCKS % 2) == 0) ? 8 : 0;
    const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
    // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
    const uint smem_rows_per_warp      = M_BLOCKS << 4;
    const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
    const uint SKEW_HALF_ACC           = ((M_BLOCKS % 2) == 0) ? 8 : 0;
    const uint SMEM_STRIDE_ACC         = (M_BLOCKS * 16 + SKEW_HALF_ACC);
    const uint smem_elems_per_warp_acc
        = M_BLOCKS * kTileDim * SMEM_STRIDE_ACC * 2; // output in FP32
    const uint smem_elems_per_warp = (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
                                         ? smem_elems_per_warp_mat
                                         : smem_elems_per_warp_acc;
    uint       output_size         = num_cols + (num_rows * (num_rows - 1) >> 1) + pad;

    bool float4_predicate = !((num_cols & 7) || (output_size & 7));

    // aligned case
    if(float4_predicate)
    {
        if(amp_train)
        {
            hipLaunchKernelGGL(
                (dotBasedInteractFwdKernel<warps_per_threadblock,
                                           threadblock_size,
                                           M_BLOCKS,
                                           K_BLOCKS,
                                           SMEM_STRIDE,
                                           SMEM_STRIDE_ACC,
                                           kWarpSize,
                                           kWarpSizeLog2,
                                           kTileDim,
                                           kTileDimLog2>),
                dim3((batch_size + warps_per_threadblock - 1) / warps_per_threadblock),
                dim3(threadblock_size),
                warps_per_threadblock * smem_elems_per_warp * sizeof(__half),
                0,
                (const __half*)input,
                (half*)output,
                batch_size,
                num_rows,
                num_cols,
                num_rows_after_padding,
                num_cols_after_padding,
                smem_elems_per_warp,
                smem_rows_per_warp,
                output_size,
                num_row_steps,
                num_col_steps,
                pad);
        }
        else
        {
#ifdef __HIP_PLATFORM_HCC__
            hipLaunchKernelGGL(
                (dotBasedInteractFwdKernel<warps_per_threadblock,
                                           threadblock_size,
                                           M_BLOCKS,
                                           K_BLOCKS,
                                           SMEM_STRIDE,
                                           SMEM_STRIDE_ACC,
                                           kWarpSize,
                                           kWarpSizeLog2,
                                           kTileDim,
                                           kTileDimLog2>),
                dim3((batch_size + warps_per_threadblock - 1) / warps_per_threadblock),
                dim3(threadblock_size),
                warps_per_threadblock * smem_elems_per_warp * sizeof(float),
                0,
                (const float*)input,
                (float*)output,
                batch_size,
                num_rows,
                num_cols,
                num_rows_after_padding,
                num_cols_after_padding,
                smem_elems_per_warp,
                smem_rows_per_warp,
                output_size,
                num_row_steps,
                num_col_steps,
                pad);
#else
            throw std::invalid_argument("CUDA WMMA does not support FP32.");
#endif
        }
    }
    else
    {
        if(amp_train)
        {
            hipLaunchKernelGGL(
                (dotBasedInteractFwdKernelNonAligned<warps_per_threadblock,
                                                     threadblock_size,
                                                     M_BLOCKS,
                                                     K_BLOCKS,
                                                     SMEM_STRIDE,
                                                     SMEM_STRIDE_ACC,
                                                     kWarpSize,
                                                     kWarpSizeLog2,
                                                     kTileDim,
                                                     kTileDimLog2>),
                dim3((batch_size + warps_per_threadblock - 1) / warps_per_threadblock),
                dim3(threadblock_size),
                warps_per_threadblock * smem_elems_per_warp * sizeof(__half),
                0,
                (const __half*)input,
                (half*)output,
                batch_size,
                num_rows,
                num_cols,
                num_rows_after_padding,
                num_cols_after_padding,
                smem_elems_per_warp,
                smem_rows_per_warp,
                output_size,
                num_row_steps,
                num_col_steps,
                pad);
        }
        else
        {
#ifdef __HIP_PLATFORM_HCC__
            hipLaunchKernelGGL(
                (dotBasedInteractFwdKernelNonAligned<warps_per_threadblock,
                                                     threadblock_size,
                                                     M_BLOCKS,
                                                     K_BLOCKS,
                                                     SMEM_STRIDE,
                                                     SMEM_STRIDE_ACC,
                                                     kWarpSize,
                                                     kWarpSizeLog2,
                                                     kTileDim,
                                                     kTileDimLog2>),
                dim3((batch_size + warps_per_threadblock - 1) / warps_per_threadblock),
                dim3(threadblock_size),
                warps_per_threadblock * smem_elems_per_warp * sizeof(float),
                0,
                (const float*)input,
                (float*)output,
                batch_size,
                num_rows,
                num_cols,
                num_rows_after_padding,
                num_cols_after_padding,
                smem_elems_per_warp,
                smem_rows_per_warp,
                output_size,
                num_row_steps,
                num_col_steps,
                pad);
#else
            throw std::invalid_argument("CUDA WMMA does not support FP32.");
#endif
        }
    }
}

template <uint kTileDim>
inline void dotBasedInteractBwd(void* input,
                                void* upstream_grad,
                                void* grad,
                                void* bottom_mlp_grad,
                                uint  batch_size,
                                uint  num_rows,
                                uint  num_cols,
                                uint  pad,
                                bool  amp_train)
{
#ifdef __HIP_PLATFORM_HCC__
    const uint kWarpSize = 64;
#else
    const uint kWarpSize = 32;
#endif
    const uint kWarpSizeLog2      = Log2<kWarpSize>::value;
    const uint kTileDimLog2       = Log2<kTileDim>::value;
    const uint mem_skew_size      = 8;
    const uint kWarpsPerBlock     = 128 / kWarpSize;
    const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
    const uint kNumThreads        = kWarpsPerBlock * kWarpSize;
    const uint kRowTilesPerStep   = 32 / kTileDim;
    const uint kColTilesPerStep   = 1;

    uint input_dbytes = amp_train ? sizeof(half) : sizeof(float);

    uint row_tiles_per_step = num_rows > kTileDim ? kRowTilesPerStep : 1;

    // num tiles
    uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
    uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

    // number of rows and columns after padding
    uint num_rows_after_padding = num_row_tiles << kTileDimLog2;
    uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

    // 2D ugrad size and stride
    uint interaction_ugrad_2D_stride     = num_rows_after_padding + mem_skew_size;
    uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;
    uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * input_dbytes;

    // 1D ugrad size
    uint interaction_ugrad_size              = num_rows * (num_rows - 1) >> 1;
    uint interaction_ugrad_size_with_padding = interaction_ugrad_size + pad;

    // in_out place size and stride
    uint input_stride     = num_cols_after_padding + mem_skew_size;
    uint input_size_elems = num_rows_after_padding * input_stride;
    uint input_size_bytes = input_size_elems * input_dbytes;

    // sample size
    uint sample_size = num_rows * num_cols;

    // output size
    uint output_size_elems = kTileDim * kTileDim * kRowTilesPerStep * kColTilesPerStep;
    uint output_size_bytes = output_size_elems * sizeof(float);

    // staging area size
    uint staging_area_size_bytes = output_size_bytes > interaction_ugrad_2D_size_bytes
                                       ? output_size_bytes
                                       : interaction_ugrad_2D_size_bytes;

    // Shared memory size
    uint wmma_smem_byte = num_rows_after_padding * num_rows_after_padding * input_dbytes;
    uint shared_mem_per_warp_size_byte
        = input_size_bytes + staging_area_size_bytes + wmma_smem_byte;
    uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

    uint num_blocks    = (batch_size + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
    uint num_row_steps = num_row_tiles / row_tiles_per_step;
    uint num_col_steps = num_col_tiles / kColTilesPerStep;

    bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (num_cols & 7));
    if(float4_predicate)
    {
        if(amp_train)
        {
            hipLaunchKernelGGL((dotBasedInteractBwdKernel<kWarpsPerBlock,
                                                          kNumThreads,
                                                          kRowTilesPerStep,
                                                          kColTilesPerStep,
                                                          kWarpSize,
                                                          kWarpSizeLog2,
                                                          kTileDim,
                                                          kTileDimLog2>),
                               dim3(num_blocks),
                               dim3(kNumThreads),
                               shared_mem_size_bytes,
                               0,
                               (const half*)input,
                               (const half*)upstream_grad,
                               (half*)grad,
                               (half*)bottom_mlp_grad,
                               batch_size,
                               num_rows,
                               num_cols,
                               num_rows_after_padding,
                               num_cols_after_padding,
                               sample_size,
                               interaction_ugrad_size,
                               interaction_ugrad_size_with_padding,
                               interaction_ugrad_2D_size_elems,
                               interaction_ugrad_2D_stride,
                               input_size_elems,
                               input_stride,
                               num_row_steps,
                               num_col_steps,
                               row_tiles_per_step,
                               shared_mem_per_warp_size_byte);
        }
        else
        {
#ifdef __HIP_PLATFORM_HCC__
            hipLaunchKernelGGL((dotBasedInteractBwdKernel<kWarpsPerBlock,
                                                          kNumThreads,
                                                          kRowTilesPerStep,
                                                          kColTilesPerStep,
                                                          kWarpSize,
                                                          kWarpSizeLog2,
                                                          kTileDim,
                                                          kTileDimLog2>),
                               dim3(num_blocks),
                               dim3(kNumThreads),
                               shared_mem_size_bytes,
                               0,
                               (const float*)input,
                               (const float*)upstream_grad,
                               (float*)grad,
                               (float*)bottom_mlp_grad,
                               batch_size,
                               num_rows,
                               num_cols,
                               num_rows_after_padding,
                               num_cols_after_padding,
                               sample_size,
                               interaction_ugrad_size,
                               interaction_ugrad_size_with_padding,
                               interaction_ugrad_2D_size_elems,
                               interaction_ugrad_2D_stride,
                               input_size_elems,
                               input_stride,
                               num_row_steps,
                               num_col_steps,
                               row_tiles_per_step,
                               shared_mem_per_warp_size_byte);
#else
            throw std::invalid_argument("CUDA WMMA does not support FP32.");
#endif
        }
    }
    else
    {
        if(amp_train)
        {
            hipLaunchKernelGGL((dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock,
                                                                    kNumThreads,
                                                                    kRowTilesPerStep,
                                                                    kColTilesPerStep,
                                                                    kWarpSize,
                                                                    kWarpSizeLog2,
                                                                    kTileDim,
                                                                    kTileDimLog2>),
                               dim3(num_blocks),
                               dim3(kNumThreads),
                               shared_mem_size_bytes,
                               0,
                               (const half*)input,
                               (const half*)upstream_grad,
                               (half*)grad,
                               (half*)bottom_mlp_grad,
                               batch_size,
                               num_rows,
                               num_cols,
                               num_rows_after_padding,
                               num_cols_after_padding,
                               sample_size,
                               interaction_ugrad_size,
                               interaction_ugrad_size_with_padding,
                               interaction_ugrad_2D_size_elems,
                               interaction_ugrad_2D_stride,
                               input_size_elems,
                               input_stride,
                               num_row_steps,
                               num_col_steps,
                               row_tiles_per_step,
                               shared_mem_per_warp_size_byte);
        }
        else
        {
#ifdef __HIP_PLATFORM_HCC__
            hipLaunchKernelGGL((dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock,
                                                                    kNumThreads,
                                                                    kRowTilesPerStep,
                                                                    kColTilesPerStep,
                                                                    kWarpSize,
                                                                    kWarpSizeLog2,
                                                                    kTileDim,
                                                                    kTileDimLog2>),
                               dim3(num_blocks),
                               dim3(kNumThreads),
                               shared_mem_size_bytes,
                               0,
                               (const float*)input,
                               (const float*)upstream_grad,
                               (float*)grad,
                               (float*)bottom_mlp_grad,
                               batch_size,
                               num_rows,
                               num_cols,
                               num_rows_after_padding,
                               num_cols_after_padding,
                               sample_size,
                               interaction_ugrad_size,
                               interaction_ugrad_size_with_padding,
                               interaction_ugrad_2D_size_elems,
                               interaction_ugrad_2D_stride,
                               input_size_elems,
                               input_stride,
                               num_row_steps,
                               num_col_steps,
                               row_tiles_per_step,
                               shared_mem_per_warp_size_byte);
#else
            throw std::invalid_argument("CUDA WMMA does not support FP32.");
#endif
        }
    }
}

#endif // DLRM_DOT_H
