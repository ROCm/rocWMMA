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
#ifndef WMMA_LOCAL_STORE_H
#define WMMA_LOCAL_STORE_H

#warning "LocalStore is deprecated. Please use OpaqueStore."

#include "Layout.h"
#include "Types.h"

template <typename T, uint32_t ElementsPerThread>
struct amdgcn_local_store;

template <>
struct amdgcn_local_store<float16_t, 1>
{
    using StoreT = VRegF16x1;
    __device__ static inline void exec(float16_t* localPtr, StoreT const& data, index_t offset)
    {
        localPtr[offset] = *data;
    }
};

template <>
struct amdgcn_local_store<float32_t, 1>
{
    using StoreT = VRegF32x1;
    __device__ static inline void exec(float32_t* localPtr, StoreT const& data, index_t offset)
    {
        localPtr[offset] = *data;
    }
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_local_store_dword_DxK
{
    using Config     = LocalConfig<MatrixT, DataLayout>;
    using TraitsBase = amdgcn_io_traits<BlockDim, BlockK, DataT, Config::ElementsPerThread>;

    struct Traits : public TraitsBase
    {
        using LayoutT = typename Config::template LayoutT<BlockDim, BlockK, DataT>;

        // These traits are per-load
        using Storer = amdgcn_local_store<DataT, Config::ElementsPerThread>;
        using StoreT = typename Storer::StoreT;
        using InputT = VecT<DataT, TraitsBase::UnpackedRegisterCount>;
    };

    __device__ static void
        exec(DataT* localPtr, typename Traits::InputT const& incoming, uint32_t ldm)
    {
        // Extract traits
        using Storer  = typename Traits::Storer;
        using StoreT  = typename Traits::StoreT;
        using LayoutT = typename Traits::LayoutT;

        // Arrange wave threads to starting data offsets due to layout.
        auto baseOffset = LayoutT::baseDataOffset(ldm);

        auto it = incoming.template begin<StoreT::size()>();
        static_assert(decltype(it)::Range == Traits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < Traits::IOCount; ++i)
        {
            Storer::exec(localPtr, *it, baseOffset);
            it++;
            baseOffset += LayoutT::dataOffsetIncrement(i, ldm);
        }
    }
};

#endif // WMMA_LOCAL_STORE_H
