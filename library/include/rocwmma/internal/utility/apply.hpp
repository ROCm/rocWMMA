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

#ifndef ROCWMMA_UTILITY_APPLY_HPP
#define ROCWMMA_UTILITY_APPLY_HPP

#include "../types.hpp"

namespace rocwmma
{
    // Use custom rocwmma apply implementation for cases where STL is not available
#ifndef __HIPCC_RTC__
    using std::apply;
#endif // __HIPCC_RTC__

    namespace detail
    {
        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        ROCWMMA_HOST_DEVICE constexpr decltype(auto)
            apply_impl(F fn, HIP_vector_type<DataT, Rank>& v, detail::index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }

        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        ROCWMMA_HOST_DEVICE constexpr decltype(auto)
            apply_impl(F fn, HIP_vector_type<DataT, Rank> const& v, detail::index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }

        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        ROCWMMA_HOST_DEVICE constexpr decltype(auto)
            apply_impl(F fn, non_native_vector_base<DataT, Rank> & v, detail::index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }

        template <typename F, typename DataT, uint32_t Rank, size_t... I>
        ROCWMMA_HOST_DEVICE constexpr decltype(auto)
            apply_impl(F fn, non_native_vector_base<DataT, Rank> const& v, detail::index_sequence<I...>)
        {
            return fn(get<I>(v)...);
        }
    } // namespace detail

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, HIP_vector_type<DataT, Rank>& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, HIP_vector_type<DataT, Rank> const& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, HIP_vector_type<DataT, Rank>&& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, non_native_vector_base<DataT, Rank>& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, non_native_vector_base<DataT, Rank> const& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

    template <typename F, typename DataT, uint32_t Rank>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) apply(F fn, non_native_vector_base<DataT, Rank>&& v)
    {
        constexpr std::size_t size = VecTraits<std::decay_t<decltype(v)>>::size();
        return detail::apply_impl(fn, v, detail::make_index_sequence<size>());
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_APPLY_HPP
