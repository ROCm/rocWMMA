#ifndef WMMA_BUFFER_STORE_H
#define WMMA_BUFFER_STORE_H

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "Constants.h"
#include "Layout.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ void __llvm_amdgcn_buffer_store_f32(float    vdata,
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
struct amdgcn_buffer_store<float32_t, 4>
{
    using StoreT = VRegF32x4;
    __device__ static inline void
        exec(StoreT data, v4_i32_t rsrc, index_t vindex, index_t offset, bool glc = 0, bool slc = 0)
    {
        return __llvm_amdgcn_buffer_store_f32x4(*data, rsrc, vindex, offset, glc, slc);
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
        // These traits are per-load
        using Storer = amdgcn_buffer_store<DataT, Config::ElementsPerThread>;
        using StoreT  = typename Storer::StoreT;
        static_assert(std::is_same< VecT<DataT, TraitsBase::RegistersPerIO>, StoreT>::value, "Unexpected StoreT");

        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;
        
        // Output format for entire block.
        // WMMA will load packed results.
        using InputT = VecT<DataT, TraitsBase::PackedRegisterCount>;
    };

    __device__ static void exec(typename Traits::InputT const& incoming, DataT const* data, uint32_t ldm)
    {
        // Extract traits
        using Storer  = typename Traits::Storer;
        using StoreT  = typename Traits::StoreT;
        using InputT = typename Traits::InputT;
        using LayoutT = typename Traits::LayoutT;

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
            StoreT chunk;
#pragma unroll
            for(uint32_t j = 0; j < Traits::RegistersPerIO; ++j)
            {
                chunk[j] = incoming[i * Traits::RegistersPerIO + j];
            }
           
            Storer::exec(chunk,
                         *(srd), // SRD regs
                         0, // stride offset
                         (initOffset + LayoutT::iterativeOffset(i, ldm)) * sizeof(DataT), // offset bytes
                         false,
                         false);
        }
    }
};

#endif // WMMA_BUFFER_STORE_H
