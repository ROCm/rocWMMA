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
#ifndef WMMA_BUFFER_STORE_H
#define WMMA_BUFFER_STORE_H

#warning "BufferStore is deprecated. Please use OpaqueStore"

#include <hip/hip_runtime.h>

#include "BufferDescriptor.h"
#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"
#include "Utils.h"

// Declare LLVM IR hook
__device__ void __llvm_amdgcn_buffer_store_f16(float16_t vdata,
                                               v4_i32_t  rsrc,
                                               index_t   vindex,
                                               index_t   offset,
                                               bool      glc,
                                               bool      slc) __asm("llvm.amdgcn.buffer.store.f16");

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
                                               v4_i32_t  rsrc,
                                               index_t   vindex,
                                               index_t   offset,
                                               bool      glc,
                                               bool      slc) __asm("llvm.amdgcn.buffer.store.f32");

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

// Buffer store dword meta-data
template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t ElementsPerThread>
struct amdgcn_buffer_store_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;

    struct Traits
    {
        // Matrix space thread offsets
        using LayoutT = StoreLayout<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;

        // These traits are per-load
        using Storer = amdgcn_buffer_store<DataT, ElementsPerThread>;
        using StoreT = typename Storer::StoreT;
        using InputT = VecT<DataT, IOTraits::UnpackedRegisterCount>;
    };

    __device__ static void exec(DataT* data, typename Traits::InputT const& incoming, uint32_t ldm)
    {
        // Extract traits
        using LayoutT = typename Traits::LayoutT;
        using Storer  = typename Traits::Storer;
        using StoreT  = typename Traits::StoreT;

        // Address and offset calcs for each wave
        BufferDescriptor<DataT> srd(data);

        // Arrange wave threads to starting data offsets due to layout.
        auto baseOffset = LayoutT::baseDataOffset(ldm);

        auto it = incoming.template begin<StoreT::size()>();
        static_assert(decltype(it)::Range == IOTraits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
        {
            Storer::exec(*it,
                         *(srd), // SRD regs
                         0, // stride offset
                         baseOffset * sizeof(DataT), // offset bytes
                         false,
                         false);
            it++;
            baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
        }
    }
};

#endif // WMMA_BUFFER_STORE_H
