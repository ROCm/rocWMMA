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

// Buffer store dword meta-data
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct amdgcn_buffer_store_dword_traits;

template <uint32_t BlockDim, uint32_t BlockK>
struct amdgcn_buffer_store_dword_traits<BlockDim, BlockK, float32_t>
{
    using DataT  = float32_t; // Float data
    using Storer = amdgcn_buffer_store<DataT, 1>; // Load DWORD, one float per thread
    using StoreT = typename Storer::StoreT; // Output register type per load

    enum : uint32_t
    {
        StridesPerStore
        = AMDGCN_WAVE_SIZE / BlockDim, // Number of consecutive strides of BlockDim per store
        StoreCount = ceilDiv(BlockDim * BlockK,
                             AMDGCN_WAVE_SIZE) // Number of store required for BlockDim * BlockK
    };

    using ResultT = VecT<DataT, StoreCount>; // Collection of registers for total load
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_buffer_store_dword_DxK
{
    // Extend traits for WMMA purposes with extra geometric
    // layout specification coming from the MatrixT
    struct Traits : public amdgcn_buffer_store_dword_traits<BlockDim, BlockK, DataT>
    {
        using LayoutT = typename Layout::template KLayout<
            MatrixT>::template LayoutT<BlockDim, BlockK, DataT, DataLayout>;
    };

    // Extract traits
    using Storer  = typename Traits::Storer;
    using StoreT  = typename Traits::StoreT;
    using ResultT = typename Traits::ResultT;
    using LayoutT = typename Traits::LayoutT;

    __device__ static void exec(ResultT const& incoming, DataT const* data, uint32_t ldm)
    {
        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        uint32_t initOffset = LayoutT::initialOffset(ldm);

        // Loop over loads to fill BlockDim * BlockK for each wave.
        ResultT result;
#pragma unroll
        for(unsigned i = 0; i < Traits::StoreCount; i++)
        {
            Storer::exec(*(incoming[i]) * (srd), // SRD regs
                         0, // stride offset
                         (initOffset + LayoutT::iterativeOffset(i, ldm))
                             * sizeof(DataT), // offset bytes
                         false,
                         false);
        }
    }
};

#endif // WMMA_BUFFER_STORE_H
