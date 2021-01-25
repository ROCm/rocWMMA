#ifndef WMMA_BUFFER_LOAD_H
#define WMMA_BUFFER_LOAD_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Constants.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ float __llvm_amdgcn_buffer_load_f32(v4_i32_t rsrc,
                                               index_t  vindex,
                                               index_t  offset,
                                               bool     glc,
                                               bool     slc) __asm("llvm.amdgcn.buffer.load.f32");

// Basic instruction wrapper
// Buffer load doesn't have clang __builtin, so we will have to use LLVM
template <typename T, uint32_t ElementsPerThread>
struct amdgcn_buffer_load;

template <>
struct amdgcn_buffer_load<float32_t, 1>
{
    using LoadT = VRegF32x1;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f32(rsrc, vindex, offset, glc, slc));
    }
};

using amdgcn_buffer_load_f32x1 = amdgcn_buffer_load<float32_t, 1>;

// Buffer load dword meta-data
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_buffer_load_dword_traits;

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_traits<BlockDim, BlockK, float32_t>
{
    using DataT  = float32_t; // Float data
    using Loader = amdgcn_buffer_load<DataT, 1>; // Load DWORD, one float per thread
    using LoadT  = typename Loader::LoadT; // Output register type per load

    enum : uint32_t
    {
        StridesPerLoad
        = AMDGCN_WAVE_SIZE / BlockDim, // Number of consecutive strides of BlockDim per load
        LoadCount = ceilDiv(BlockDim * BlockK,
                            AMDGCN_WAVE_SIZE) // Number of loads required for BlockDim * BlockK
    };

    using ResultT = VecT<DataT, LoadCount>; // Collection of registers for total load
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename Layout>
struct amdgcn_buffer_load_dword_DxK;

// Define the matrix_a layout as loading columns from each block in the K direction
template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_DxK<matrix_a, BlockDim, BlockK, float32_t, row_major>
{
    // Extract traits
    using DataT   = float32_t;
    using Traits  = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;
    using Loader  = typename Traits::Loader;
    using LoadT   = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        uint32_t rowOffset = (threadIdx.x % BlockDim) * ldm;
        uint32_t colOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad; // K Id

        // Loop over loads to fill BlockM * BlockK for each wave.
        ResultT result;
#pragma unroll
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult = Loader::exec(*(srd), // SRD regs
                                            0, // stride offset (row)
                                            (i * Traits::StridesPerLoad + colOffset + rowOffset)
                                                * sizeof(DataT), // offset bytes (col)
                                            false,
                                            false);
            result[i]        = *(loadResult);
        }
        return result;
    }
};

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_DxK<matrix_a, BlockDim, BlockK, float32_t, col_major>
{
    // Extract traits
    using DataT   = float32_t;
    using Traits  = amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>;
    using Loader  = typename Traits::Loader;
    using LoadT   = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        uint32_t rowOffset = (threadIdx.x % BlockDim);
        uint32_t colOffset = (threadIdx.x / BlockDim) % Traits::StridesPerLoad * ldm; // K Id

        // Loop over loads to fill BlockDim * BlockK for each wave.
        ResultT result;
#pragma unroll
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult
                = Loader::exec(*(srd), // SRD regs
                               0, // stride offset (row)
                               (i * Traits::StridesPerLoad * ldm + colOffset + rowOffset)
                                   * sizeof(DataT), // offset bytes (col)
                               false,
                               false);
            result[i] = *(loadResult);
        }
        return result;
    }
};

// matrix_b is simply the transpose of matrix_a
// layout as loading rows from each block in the K direction
template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_DxK<matrix_b, BlockDim, BlockK, float32_t, row_major>
    : public amdgcn_buffer_load_dword_DxK<matrix_a, BlockDim, BlockK, float32_t, col_major>
{
};

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_DxK<matrix_b, BlockDim, BlockK, float32_t, col_major>
    : public amdgcn_buffer_load_dword_DxK<matrix_a, BlockDim, BlockK, float32_t, row_major>
{
};

#endif // WMMA_BUFFER_LOAD_H
