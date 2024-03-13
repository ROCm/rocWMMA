/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_TRANSFORMS_HPP
#define ROCWMMA_TRANSFORMS_HPP

#include "vector.hpp"

namespace rocwmma
{
    template <uint32_t BlockDim, uint32_t VW>
    struct AosToSoa;

    template <uint32_t BlockDim, uint32_t VW>
    struct SoaToAos;

    template <uint32_t BlockDim, uint32_t MaxVW, typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto soa_to_aos(VecT<DataT, VecSize> const& v);

    template <uint32_t BlockDim, uint32_t MaxVW, typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto aos_to_soa(VecT<DataT, VecSize> const& v);

} // namespace rocwmma

#include "transforms_impl.hpp"

#endif // ROCWMMA_TRANSFORMS_HPP
