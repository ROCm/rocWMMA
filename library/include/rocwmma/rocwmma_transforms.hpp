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
#ifndef ROCWMMA_TRANSFORMS_API_HPP
#define ROCWMMA_TRANSFORMS_API_HPP

#include "rocwmma.hpp"
#include "rocwmma_transforms_impl.hpp"

namespace rocwmma
{
    // Implicit transpose of fragment
    template <typename FragT>
    using ApplyTranspose_t = typename detail::template ApplyTranspose<FragT>::Type;

    // Implicit layout change of fragment
    template <typename FragT, typename NewDataLayoutT>
    using ApplyDataLayout_t =
        typename detail::template ApplyDataLayout<FragT, NewDataLayoutT>::Type;

    template <typename FragT>
    using ApplyRegisterFile_t = typename detail::template ApplyRegisterFile<FragT>::Type;

    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) applyTranspose(FragT&& frag);

    template <typename DataLayoutT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) applyDataLayout(FragT&& frag);

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_API_HPP
