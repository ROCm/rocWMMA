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

#ifndef ROCWMMA_UTILITY_VECTOR_HPP
#define ROCWMMA_UTILITY_VECTOR_HPP

#include "vector_impl.hpp"

namespace rocwmma
{
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr inline auto vector_size(VecT const& v);

    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) make_vector(Ts&&... ts);

    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) vector_cat(Lhs&& lhs, Rhs&& rhs);

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
        vector_reduce_and(VecT&& lhs) noexcept;

    template <typename DataT>
    ROCWMMA_HOST_DEVICE constexpr inline auto swap(HIP_vector_type<DataT, 2> const& v);
} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_HPP
