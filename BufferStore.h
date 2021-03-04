#ifndef WMMA_BUFFER_STORE_H
#define WMMA_BUFFER_STORE_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Constants.h"
#include "IOConfig.h"
#include "IOUnpack.h"
#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ void __llvm_amdgcn_buffer_store_f16(float16_t    vdata,
                                               v4_i32_t rsrc,
                                               index_t  vindex,
                                               index_t  offset,
                                               bool     glc,
                                               bool     slc) __asm("llvm.amdgcn.buffer.store.f16");

__device__ void __llvm_amdgcn_buffer_store_f16x2(v2_f16_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t  vindex,
                                                 index_t  offset,
                                                 bool     glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f16");

__device__ void __llvm_amdgcn_buffer_store_f16x4(v4_f16_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t  vindex,
                                                 index_t  offset,
                                                 bool     glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f16");

__device__ void __llvm_amdgcn_buffer_store_f32(float32_t vdata,
                                               v4_i32_t rsrc,
                                               index_t  vindex,
                                               index_t  offset,
                                               bool     glc,
                                               bool     slc) __asm("llvm.amdgcn.buffer.store.f32");

__device__ void __llvm_amdgcn_buffer_store_f32x2(v2_f32_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t  vindex,
                                                 index_t  offset,
                                                 bool     glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

__device__ void __llvm_amdgcn_buffer_store_f32x4(v4_f32_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t  vindex,
                                                 index_t  offset,
                                                 bool     glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f32");

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_buffer_store;

template <>
struct amdgcn_buffer_store<float16_t, 1>
{
    using StoreT = VRegF16x1;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f16(*data, rsrc, vindex, offset, glc, slc);
    }
};

template <>
struct amdgcn_buffer_store<float16_t, 2>
{
    using StoreT = VRegF16x2;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f16x2(*data, rsrc, vindex, offset, glc, slc);
    }
};

template <>
struct amdgcn_buffer_store<float16_t, 4>
{
    using StoreT = VRegF16x4;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f16x4(*data, rsrc, vindex, offset, glc, slc);
    }
};



template <>
struct amdgcn_buffer_store<float32_t, 1>
{
    using StoreT = VRegF32x1;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f32(*data, rsrc, vindex, offset, glc, slc);
    }
};

template <>
struct amdgcn_buffer_store<float32_t, 2>
{
    using StoreT = VRegF32x2;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f32x2(*data, rsrc, vindex, offset, glc, slc);
    }
};

template <>
struct amdgcn_buffer_store<float32_t, 4>
{
    using StoreT = VRegF32x4;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f32x4(*data, rsrc, vindex, offset, glc, slc);
    }
};

template<typename T, uint32_t ElementsPerThread>
struct amdgcn_opaque_store;

template <>
struct amdgcn_opaque_store<float16_t, 1>
{
    using StoreT = VRegF16x1;
    __device__ static inline void exec(float16_t* ptr, StoreT const& data, index_t offset) 
    {
        ptr[offset] = *data;
    }
};

template <>
struct amdgcn_opaque_store<float32_t, 1>
{
    using StoreT = VRegF32x1;
    __device__ static inline void exec(float32_t* ptr, StoreT const& data, index_t offset) 
    {
        ptr[offset] = *data;
    }
};


// Buffer store dword meta-data
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_buffer_store_dword_DxK
{
    using Config = BufferConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;
    
    struct Traits : public TraitsBase
    {
        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;

        // These traits are per-load
        using Storer = amdgcn_buffer_store<DataT, Config::ElementsPerThread>;
        using StoreT  = typename Storer::StoreT;
        using InputT = VecT<DataT, TraitsBase::UnpackedRegisterCount>;
    };

    __device__ static void exec(DataT* data, typename Traits::InputT const& incoming, uint32_t ldm)
    {
        // Extract traits
        using LayoutT = typename Traits::LayoutT;
        using Storer  = typename Traits::Storer;
        using StoreT = typename Traits::StoreT;
        
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        auto it = incoming.template begin<StoreT::size()>();
        static_assert(decltype(it)::Range == Traits::IOCount, "IOCount inconsistent with iterator range");
        
#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
#ifdef F16_LLVM_BUG_WORKAROUND
            amdgcn_opaque_store<DataT, 1>::exec(
                        data, *it, initOffset + LayoutT::iterativeOffset(i, ldm));
#else    
            Storer::exec(*it,
                         *(srd), // SRD regs
                         0, // stride offset
                         (initOffset + LayoutT::iterativeOffset(i, ldm)) * sizeof(DataT), // offset bytes
                         false,
                         false);
#endif // F16_LLVM_BUG_WORKAROUND
            it++;
        }
    }
};

#endif // WMMA_BUFFER_STORE_H
