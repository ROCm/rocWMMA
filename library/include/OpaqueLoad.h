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
#ifndef WMMA_OPAQUE_LOAD_H
#define WMMA_OPAQUE_LOAD_H

#include "IOTraits.h"
#include "Layout.h"
#include "MappingUtil.h"
#include "Types.h"
#include "Utils.h"

namespace rocwmma
{

    namespace detail
    {

        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_opaque_load
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");

            using LoadT = VecT<typename PackTraits<DataT>::UnpackedT, VectorWidth>;
            __device__ static inline auto exec(DataT const* dataPtr, index_t offset = 0) -> LoadT
            {
                return LoadT(
                    *reinterpret_cast<typename LoadT::StorageT const*>(&(dataPtr[offset])));
            }
        };

    } // namespace detail

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataMapper,
              class MatrixMapper,
              uint32_t VectorWidth>
    struct OpaqueLoad
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

        struct Traits
        {
            // Raw IO on unpacked register data.
            using Loader  = detail::amdgcn_opaque_load<DataT, VectorWidth>;
            using LoadT   = typename Loader::LoadT;
            using OutputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        __device__ static auto exec(DataT const* localPtr, uint32_t ldm) -> typename Traits::OutputT
        {
            // Extract traits
            using Loader  = typename Traits::Loader;
            using LoadT   = typename Traits::LoadT;
            using OutputT = typename Traits::OutputT;

            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset = MatrixMapper::baseOffset();

            OutputT result;
            auto    it = result.template begin<LoadT::size()>();

            static_assert(decltype(it)::range() == IOTraits::IOCount,
                          "IOCount inconsistent with iterator range");

            // Loop through entire block
#pragma unroll
            for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
            {
                *it = *Loader::exec(localPtr, DataMapper::fromMatrixCoord(baseOffset, ldm));
                it++;
                baseOffset += MatrixMapper::incrementalOffset(i);
            }
            return result;
        }
    };

} // namespace rocwmma

#endif // WMMA_OPAQUE_LOAD_H
