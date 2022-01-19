/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef WMMA_OPAQUE_STORE_H
#define WMMA_OPAQUE_STORE_H

#include "IOTraits.h"
#include "Layout.h"
#include "Types.h"

template <typename DataT, uint32_t VectorWidth>
struct amdgcn_opaque_store
{
    static_assert(VectorWidth > 0, "Vector width must be greater than 0");

    using StoreT = VecT<typename PackTraits<DataT>::UnpackedT, VectorWidth>;
    __device__ static inline void exec(DataT* dataPtr, StoreT const& data, index_t offset = 0)
    {
        *reinterpret_cast<typename StoreT::StorageT*>(&(dataPtr[offset])) = *data;
    }
};

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class StoreLayout,
          uint32_t VectorWidth>
struct amdgcn_opaque_store_DxK
{
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;

    struct Traits
    {
        // Matrix space thread offsets
        using MatrixLayout = StoreLayout<BlockDim, BlockK, DataT, DataLayout, VectorWidth>;
        using MappingUtil  = typename MatrixLayout::Traits::MappingUtil;

        // Raw IO on unpacked register data.
        using Storer = amdgcn_opaque_store<DataT, VectorWidth>;
        using StoreT = typename Storer::StoreT;
        using InputT = VecT<DataT, IOTraits::UnpackedSize>;
    };

    __device__ static void
        exec(DataT* localPtr, typename Traits::InputT const& incoming, uint32_t ldm)
    {
        // Extract traits
        using Storer       = typename Traits::Storer;
        using StoreT       = typename Traits::StoreT;
        using MatrixLayout = typename Traits::MatrixLayout;
        using MappingUtil  = typename Traits::MappingUtil;

        // Arrange wave threads to starting data offsets due to layout.
        // In this case, the LDS contains only block data.
        auto baseOffset = MatrixLayout::baseOffset();

        auto it = incoming.template begin<StoreT::size()>();
        static_assert(decltype(it)::range() == IOTraits::IOCount,
                      "IOCount inconsistent with iterator range");

#pragma unroll
        for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
        {
            Storer::exec(localPtr, *it, MappingUtil::dataOffset(baseOffset, ldm));
            it++;
            baseOffset += MatrixLayout::incrementalOffset(i);
        }
    }
};

#endif // WMMA_OPAQUE_STORE_H
