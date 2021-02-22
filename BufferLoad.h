#ifndef WMMA_BUFFER_LOAD_H
#define WMMA_BUFFER_LOAD_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Constants.h"
#include "IOConfig.h"
#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ float __llvm_amdgcn_buffer_load_f32(v4_i32_t rsrc,
                                               index_t  vindex,
                                               index_t  offset,
                                               bool     glc,
                                               bool     slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ v4_f32_t
__llvm_amdgcn_buffer_load_f32x4(v4_i32_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v4f32");

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

template <>
struct amdgcn_buffer_load<float32_t, 4>
{
    using LoadT = VRegF32x4;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f32x4(rsrc, vindex, offset, glc, slc));
    }
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_buffer_load_dword_DxK
{
    using Config = BufferConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;

    struct Traits : public TraitsBase
    {
        // These traits are per-load
        using Loader = amdgcn_buffer_load<DataT, Config::ElementsPerThread>;
        using LoadT  = typename Loader::LoadT;

        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;
        
        // Output format for entire block.
        // WMMA will load packed results.
        using OutputT = VecT<DataT, TraitsBase::PackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> typename Traits::OutputT
    {
        // Extract traits
        using Loader  = typename Traits::Loader;
        using LoadT  = typename Traits::LoadT;
        using OutputT = typename Traits::OutputT;
        using LayoutT = typename Traits::LayoutT;

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        OutputT result;
#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
            LoadT loadResult = Loader::exec(*(srd), // SRD regs
                                    0, // stride offset
                                    (initOffset + LayoutT::iterativeOffset(i, ldm))
                                        * sizeof(DataT), // offset bytes
                                    false,
                                    false);
#pragma unroll
            for(uint32_t j = 0; j < Traits::RegistersPerIO; ++j)
            {
                result[i*Traits::RegistersPerIO + j] = loadResult[j];
            }
        }
        return result;
    }
};

#endif // WMMA_BUFFER_LOAD_H
