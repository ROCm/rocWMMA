#ifndef WMMA_BUFFER_LOAD_H
#define WMMA_BUFFER_LOAD_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"
#include "Utils.h"

#define F16_LLVM_BUG_WORKAROUND

// Declare LLVM IR hook
__device__ float32_t __llvm_amdgcn_buffer_load_f32(v4_i32_t rsrc,
                                                   index_t  vindex,
                                                   index_t  offset,
                                                   bool     glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ v2_f32_t
    __llvm_amdgcn_buffer_load_f32x2(v4_i32_t rsrc,
                                    index_t  vindex,
                                    index_t  offset,
                                    bool     glc,
                                    bool     slc) __asm("llvm.amdgcn.buffer.load.v2f32");

__device__ v4_f32_t
    __llvm_amdgcn_buffer_load_f32x4(v4_i32_t rsrc,
                                    index_t  vindex,
                                    index_t  offset,
                                    bool     glc,
                                    bool     slc) __asm("llvm.amdgcn.buffer.load.v4f32");

__device__ float16_t __llvm_amdgcn_buffer_load_f16(v4_i32_t rsrc,
                                                   index_t  vindex,
                                                   index_t  offset,
                                                   bool     glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.f16");

__device__ v2_f16_t
    __llvm_amdgcn_buffer_load_f16x2(v4_i32_t rsrc,
                                    index_t  vindex,
                                    index_t  offset,
                                    bool     glc,
                                    bool     slc) __asm("llvm.amdgcn.buffer.load.v2f16");

__device__ v4_f16_t
    __llvm_amdgcn_buffer_load_f16x4(v4_i32_t rsrc,
                                    index_t  vindex,
                                    index_t  offset,
                                    bool     glc,
                                    bool     slc) __asm("llvm.amdgcn.buffer.load.v4f16");

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
struct amdgcn_buffer_load<float32_t, 2>
{
    using LoadT = VRegF32x2;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f32x2(rsrc, vindex, offset, glc, slc));
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

template <>
struct amdgcn_buffer_load<float16_t, 1>
{
    using LoadT = VRegF16x1;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f16(rsrc, vindex, offset, glc, slc));
    }
};

template <>
struct amdgcn_buffer_load<float16_t, 2>
{
    using LoadT = VRegF16x2;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f16x2(rsrc, vindex, offset, glc, slc));
    }
};

template <>
struct amdgcn_buffer_load<float16_t, 4>
{
    using LoadT = VRegF16x4;
    __device__ static inline auto
        exec(v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0) -> LoadT
    {
        return LoadT(__llvm_amdgcn_buffer_load_f16x4(rsrc, vindex, offset, glc, slc));
    }
};

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class LoadLayout,
          uint32_t ElementsPerThread>
struct amdgcn_buffer_load_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    struct Traits
    {
        // Matrix space thread offsets
        using LayoutT = LoadLayout<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;

        // Raw IO that produce unpacked register data.
        using Loader  = amdgcn_buffer_load<DataT, ElementsPerThread>;
        using LoadT   = typename Loader::LoadT;
        using OutputT = VecT<DataT, IOTraits::UnpackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> typename Traits::OutputT
    {
        // Extract traits
        using LayoutT = typename Traits::LayoutT;
        using Loader  = typename Traits::Loader;
        using LoadT   = typename Traits::LoadT;
        using OutputT = typename Traits::OutputT;

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        OutputT result;
        auto    it = result.template begin<LoadT::size()>();

        static_assert(decltype(it)::Range == IOTraits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
        {
            *it = *Loader::exec(*(srd), // SRD regs
                                0, // stride offset
                                (initOffset + LayoutT::iterativeOffset(i, ldm))
                                    * sizeof(DataT), // offset bytes
                                false,
                                false);
            it++;
        }

        return result;
    }
};

#endif // WMMA_BUFFER_LOAD_H
