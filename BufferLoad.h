#ifndef WMMA_BUFFER_LOAD_H
#define WMMA_BUFFER_LOAD_H


#include <hip/hip_runtime.h>


#include "BufferDescriptor.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ float __llvm_amdgcn_buffer_load_f32(v4_i32_t rsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.load.f32");

// Basic instruction wrapper
// Buffer load doesn't have clang __builtin, so we will have to use LLVM
template <typename T, uint32_t ElementsPerThread>
struct amdgcn_buffer_load;

template<>
struct amdgcn_buffer_load<float32_t, 1>
{
    using LoadT = VRegF32x1;
    __device__ static inline auto exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f32(rsrc, vindex, offset, glc, slc));
    }
};

using amdgcn_buffer_load_f32x1 = amdgcn_buffer_load<float32_t, 1>;


// Buffer load dword meta-data
template<uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_buffer_load_dword_traits;

template<uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_load_dword_traits<BlockDim, BlockK, float32_t>
{
    using DataT = float32_t;                               // Float data
    using Loader = amdgcn_buffer_load<DataT, 1>;           // Load DWORD, one float per thread
    using LoadT = typename Loader::LoadT;                  // Output register type per load
    
    enum { StridesPerLoad = 64 / BlockDim };               // Number of consecutive strides of BlockDim per load
    enum { LoadCount = ceilDiv(BlockDim*BlockK, 64) };     // Number of loads required for BlockDim * BlockK

    using ResultT = VecT<DataT, LoadCount>;                // Collection of registers for total load
};


template <typename Mat, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename Layout>
struct amdgcn_buffer_load_dword_MxNxK;

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_buffer_load_dword_MxNxK<matrix_a, BlockM, BlockN, BlockK, float32_t, row_major>
{
    // Extract traits
    using DataT = float32_t;
    using Traits = amdgcn_buffer_load_dword_traits<BlockM, BlockK, DataT>;
    using Loader = typename Traits::Loader;
    using LoadT = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data, ldm);

        uint32_t rowOffset = threadIdx.x % BlockM;
        uint32_t waveColOffset = (threadIdx.x / 64) * BlockN;                    // Wave ID
        uint32_t kColOffset = (threadIdx.x / BlockM) % Traits::StridesPerLoad;   // K Id 
        uint32_t colOffset = waveColOffset + kColOffset;
        
        // Loop over loads to fill BlockM * BlockK for each wave.
        ResultT result;
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult = Loader::exec( 
                *(srd),                                                     // SRD regs
                rowOffset,                                                  // stride offset (row)
                (i * Traits::StridesPerLoad + colOffset) * sizeof(DataT),   // offset bytes (col)
                false,
                false);
            result[i] = *(loadResult);
            //result[i] = blockIdx.y;
        }
        return result;
    }
};

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_buffer_load_dword_MxNxK<matrix_a, BlockM, BlockN, BlockK, float32_t, col_major>
{
    using DataT = float32_t;
    using Traits = amdgcn_buffer_load_dword_traits<BlockM, BlockK, DataT>;
    using Loader = typename Traits::Loader;
    using LoadT = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {
        // Move the data origin to the start of the block data.
        uint32_t startOffset = 
            (blockIdx.x * ldm * (blockDim.x / 64) * BlockM) +             // Start row
            blockIdx.y * BlockN ; // Start col

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data + startOffset, ldm);

        uint32_t colOffset = threadIdx.x % BlockN;
        uint32_t waveRowOffset = (threadIdx.x / 64) * BlockM;                    // Wave ID
        uint32_t kRowOffset = (threadIdx.x / BlockN) % Traits::StridesPerLoad;   // K Id 
        uint32_t rowOffset = waveRowOffset + kRowOffset;
        
        // Loop over loads to fill BlockM * BlockK for each wave.
        ResultT result;
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult = Loader::exec( 
                *(srd),                                     // SRD regs
                i * Traits::StridesPerLoad + rowOffset,     // stride offset (row)
                colOffset * sizeof(DataT),                  // offset bytes (col)
                false,
                false);
            result[i] = *(loadResult);
        }
        return result;
    }
};

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_buffer_load_dword_MxNxK<matrix_b, BlockM, BlockN, BlockK, float32_t, row_major>
{
    using DataT = float32_t;
    using Traits = amdgcn_buffer_load_dword_traits<BlockM, BlockK, DataT>;
    using Loader = typename Traits::Loader;
    using LoadT = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {
        // Move the data origin to the start of the block data.
        uint32_t startOffset = 
            (blockIdx.x * ldm * BlockN) +  // Start row
            blockIdx.y * blockDim.x;       // Start col

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data, ldm);

        uint32_t waveColOffset = (threadIdx.x / 64) * BlockN;                   // Wave ID
        uint32_t colOffset = threadIdx.x % BlockN + waveColOffset;
        uint32_t rowOffset = (threadIdx.x / BlockN) % Traits::StridesPerLoad;   // K Id 
        
        // Loop over loads to fill BlockM * BlockK for each wave.
        ResultT result;
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult = Loader::exec( 
                *(srd),                                     // SRD regs
                i * Traits::StridesPerLoad + rowOffset,     // stride offset (row)
                colOffset * sizeof(DataT),                  // offset bytes (col)
                false,
                false);
            result[i] = *(loadResult);
        }
        return result;
    }
};

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_buffer_load_dword_MxNxK<matrix_b, BlockM, BlockN, BlockK, float32_t, col_major>
{
    // Extract traits
    using DataT = float32_t;
    using Traits = amdgcn_buffer_load_dword_traits<BlockM, BlockK, DataT>;
    using Loader = typename Traits::Loader;
    using LoadT = typename Traits::LoadT;
    using ResultT = typename Traits::ResultT;

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> ResultT
    {

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data, ldm);

        uint32_t waveRowOffset = (threadIdx.x / 64) * BlockN;                    // Wave ID
        uint32_t rowOffset = threadIdx.x % BlockM + waveRowOffset;
        uint32_t colOffset = (threadIdx.x / BlockM) % Traits::StridesPerLoad;    // K Id 
        
        // Loop over loads to fill BlockM * BlockK for each wave.
        ResultT result;
        for(unsigned i = 0; i < Traits::LoadCount; i++)
        {
            LoadT loadResult = Loader::exec( 
                *(srd),                                                     // SRD regs
                rowOffset,                                                  // stride offset (row)
                (i * Traits::StridesPerLoad + colOffset) * sizeof(DataT),   // offset bytes (col)
                false,
                false);
            result[i] = *(loadResult);
        }
        return result;
    }
};

#endif // WMMA_BUFFER_LOAD_H
