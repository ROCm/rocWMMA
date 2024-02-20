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

#ifndef ROCWMMA_UTILITY_FORWARD_IMPL_HPP
#define ROCWMMA_UTILITY_FORWARD_IMPL_HPP

#include "type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {

        template <typename T>
        ROCWMMA_HOST_DEVICE constexpr T&& forward(typename remove_reference<T>::type& t) noexcept
        {
            return static_cast<T&&>(t);
        }

        template <typename T>
        ROCWMMA_HOST_DEVICE constexpr T&& forward(typename remove_reference<T>::type&& t) noexcept
        {
            static_assert(!is_lvalue_reference<T>::value,
                          "template argument substituting T is an lvalue reference type");
            return static_cast<T&&>(t);
        }

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_FORWARD_IMPL_HPP
