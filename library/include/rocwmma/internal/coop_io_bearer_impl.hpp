/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_COOP_IO_BEARER_IMPL_HPP
#define ROCWMMA_COOP_IO_BEARER_IMPL_HPP

#include "io_bearer.hpp"
#include "layout/matrix_coop_layout.hpp"
#include "utility/math.hpp"
#include "vector.hpp"

namespace rocwmma
{

#define CoopIOBearerTypesDecl                 \
    class DataLayout, class CoopMatrixLayout, \
        template <typename, uint32_t>         \
        class BearerPolicy, class BoundsCtrl
#define CoopIOBearerTypesImpl DataLayout, CoopMatrixLayout, BearerPolicy, BoundsCtrl

    // Forwards to iterative unroll because waveCount override may not be known at compile time.
    template <CoopIOBearerTypesDecl>
    template <typename BufferT, typename ExternDataT>
    ROCWMMA_DEVICE inline void CoopIOBearer<CoopIOBearerTypesImpl>::exec(BufferT&&    buffer,
                                                                         ExternDataT* dataPtr,
                                                                         uint32_t     ldm,
                                                                         uint32_t     waveIndex,
                                                                         uint32_t     waveCount)
    {
        // Filter out waves we don't want participating
        if(!CoopMatrixLayout::waveEnabler(waveIndex, waveCount))
        {
            return;
        }

        // The base offset will include the wave offset
        auto baseOffset = CoopMatrixLayout::baseOffset(waveIndex, waveCount);

        // Dispatch run-time wave count to the IOBearer class
        static_for<0u, 3u, 1u>(
            [](auto&& idx,
               auto   waveCount,
               auto&& buffer,
               auto&& baseOffset,
               auto*  dataPtr,
               auto   ldm) {
                // Wave counts only supported as powers of 2
                constexpr auto Idx       = pow2<decay_t<decltype(idx)>::value>::value;
                bool           processed = false;

                if(Idx == waveCount && !processed)
                {
                    // We need to shrink the buffer for the unroll
                    constexpr auto BuffSize
                        = reduce_mult(CoopMatrixLayout::strideCounts(Idx)) * TransactionSize;
                    using ReducedBufferT = conditional_t<is_const_v<BufferT>,
                                                         VecT<DataT, BuffSize> const,
                                                         VecT<DataT, BuffSize>>;

                    // Base can unroll statically
                    Base::unroll_impl((ReducedBufferT&)(buffer), baseOffset, dataPtr, ldm);

                    processed = true;
                }
            },
            waveCount,
            forward<BufferT>(buffer),
            baseOffset,
            dataPtr,
            ldm);
    }

    // Forwards to base class static unroll using static WaveCount
    template <CoopIOBearerTypesDecl>
    template <typename BufferT, typename ExternDataT>
    ROCWMMA_DEVICE inline void CoopIOBearer<CoopIOBearerTypesImpl>::exec(BufferT&&    buffer,
                                                                         ExternDataT* dataPtr,
                                                                         uint32_t     ldm,
                                                                         uint32_t     waveIndex)
    {
        // Filter out waves we don't want participating
        if(!CoopMatrixLayout::waveEnabler(waveIndex, WaveCount))
        {
            return;
        }

        // The base offset will include the wave offset
        auto baseOffset = CoopMatrixLayout::baseOffset(waveIndex, WaveCount);

        // We need to shrink the buffer for the unroll
        constexpr auto BuffSize
            = reduce_mult(CoopMatrixLayout::strideCounts(WaveCount)) * TransactionSize;
        using ReducedBufferT = conditional_t<is_const_v<BufferT>,
                                             VecT<DataT, BuffSize> const,
                                             VecT<DataT, BuffSize>>;

        // Base can unroll statically
        Base::unroll_impl((ReducedBufferT&)(buffer), baseOffset, dataPtr, ldm);
    }

#undef CoopIOBearerTypesDecl
#undef CoopIOBearerTypesImpl

} // namespace rocwmma

#endif // ROCWMMA_COOP_IO_BEARER_IMPL_HPP
