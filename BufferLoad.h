#ifndef WMMA_BUFFER_LOAD_H
#define WMMA_BUFFER_LOAD_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Constants.h"
#include "IOConfig.h"
#include "IOPack.h"
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
                                               bool     slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ v2_f32_t
__llvm_amdgcn_buffer_load_f32x2(v4_i32_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v2f32");

__device__ v4_f32_t
__llvm_amdgcn_buffer_load_f32x4(v4_i32_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v4f32");

__device__ float16_t __llvm_amdgcn_buffer_load_f16(v4_i32_t rsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.f16");

__device__ v2_f16_t __llvm_amdgcn_buffer_load_f16x2(v4_i32_t rsrc,
                                                   index_t vindex,
                                                   index_t offset,
                                                   bool glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.v2f16");

__device__ v4_f16_t __llvm_amdgcn_buffer_load_f16x4(v4_i32_t rsrc,
                                                   index_t vindex,
                                                   index_t offset,
                                                   bool glc,
                                                   bool slc) __asm("llvm.amdgcn.buffer.load.v4f16");

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

template<typename T, uint32_t ElementsPerThread>
struct amdgcn_opaque_load;

template <>
struct amdgcn_opaque_load<float16_t, 1>
{
    using LoadT = VRegF16x1;
    __device__ static inline auto exec(float16_t const* data, index_t offset) -> LoadT
    {
        return LoadT(data[offset]);
    }
};

template <>
struct amdgcn_opaque_load<float32_t, 1>
{
    using LoadT = VRegF32x1;
    __device__ static inline auto exec(float32_t const* data, index_t offset) -> LoadT
    {
        return LoadT(data[offset]);
    }
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_buffer_load_dword_DxK
{
    using Config = BufferConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;

    struct Traits : public TraitsBase
    {
         // Matrix space thread offsets
        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;

        // Raw IO that produce unpacked register data.
        using Loader = amdgcn_buffer_load<DataT, Config::ElementsPerThread>; // Intrinsic wrapper
        using LoadT  = typename Loader::LoadT; // Intrinsic output (unpacked)
        using OutputT = VecT<DataT, TraitsBase::UnpackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* data, uint32_t ldm) -> typename Traits::OutputT
    {
        // Extract traits
        using LayoutT = typename Traits::LayoutT;
        using Loader  = typename Traits::Loader;
        using LoadT = typename Traits::LoadT;
        using OutputT = typename Traits::OutputT;
        
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        OutputT result;
        auto it = result.template begin<LoadT::size()>();

        static_assert(decltype(it)::Range == Traits::IOCount, "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
#ifdef F16_LLVM_BUG_WORKAROUND
            *it = *amdgcn_opaque_load<DataT, 1>::exec(
                        data, initOffset + LayoutT::iterativeOffset(i, ldm));
#else

            *it = *Loader::exec(*(srd), // SRD regs
                                    0, // stride offset
                                    (initOffset + LayoutT::iterativeOffset(i, ldm))
                                        * sizeof(DataT), // offset bytes
                                    false,
                                    false);

#endif // F16_LLVM_BUG_WORKAROUND
            it++;
        }

        return result;
    }
};

#endif // WMMA_BUFFER_LOAD_H
