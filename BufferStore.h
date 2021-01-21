#ifndef WMMA_BUFFER_STORE_H
#define WMMA_BUFFER_STORE_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Types.h"


// Declare LLVM IR hook
__device__ void __llvm_amdgcn_buffer_store_f32(float vdata,
                                               v4_i32_t rsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.store.f32");

__device__ void __llvm_amdgcn_buffer_store_f32x2(v2_f32_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

__device__ void __llvm_amdgcn_buffer_store_f32x4(v4_f32_t vdata,
                                                 v4_i32_t rsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f32");

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_buffer_store;

template<>
struct amdgcn_buffer_store<float32_t, 1>
{
    using RegT = VRegF32x1;
    __device__ static inline void exec(RegT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f32(*data, rsrc, vindex, offset, glc, slc);
    }
};

#endif // WMMA_BUFFER_STORE_H
