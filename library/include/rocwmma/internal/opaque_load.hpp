/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_OPAQUE_LOAD_HPP
#define ROCWMMA_OPAQUE_LOAD_HPP

#include "io_traits.hpp"
#include "layout.hpp"
#include "types.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_opaque_load
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(VecT<DataT, VectorWidth>),
                          "Cannot vectorize input");

            using LoadT = VecT<DataT, VectorWidth>;

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                data = *reinterpret_cast<LoadT const*>(&(dataPtr[offset]));
            }
        };

    } // namespace detail

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
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

        using LoadVecTraits = VecTraits<typename Traits::LoadT>;

        ROCWMMA_DEVICE static void
            exec(typename Traits::OutputT& data, DataT const* dataPtr, uint32_t ldm)
        {
            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset = MatrixLayout::baseOffset();
            auto it         = makeVectorIterator<LoadVecTraits::size()>(data).begin();

            static_assert(decltype(it)::range() == IOTraits::IOCount,
                          "IOCount inconsistent with iterator range");

            // Loop through entire block
#pragma unroll
            for(uint32_t i = 0; i < IOTraits::IOCount; ++i)
            {
                Traits::Loader::exec(*it, dataPtr, DataLayout::fromMatrixCoord(baseOffset, ldm));
                baseOffset += MatrixLayout::incrementalOffset(it.index());
                it++;
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_OPAQUE_LOAD_HPP
