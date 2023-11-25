/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_TUPLE_HPP
#define ROCWMMA_TUPLE_HPP

#include <iostream>
#include <tuple>

#include "utils.hpp"

namespace rocwmma
{
    using detail::index_sequence;
    using detail::make_index_sequence;

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator+(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} += non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator+(U y, const non_native_vector_base<T, n>& x) noexcept
    {
        return non_native_vector_base<T, n>{x} += non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator-(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} -= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator-(U y, const non_native_vector_base<T, n>& x) noexcept
    {
        return non_native_vector_base<T, n>{x} -= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator*(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} *= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator*(U y, const non_native_vector_base<T, n>& x) noexcept
    {
        return non_native_vector_base<T, n>{x} *= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator/(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} /= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, n>
        operator/(U y, const non_native_vector_base<T, n>& x) noexcept
    {
        return non_native_vector_base<T, n>{x} /= non_native_vector_base<T, n>{y};
    }

    namespace detail
    {
        template <typename VecT, std::size_t... Indices>
        constexpr static auto copy_impl(VecT&& t, index_sequence<Indices...>&&)
        {
            return make_vector(std::get<Indices>(std::forward<VecT>(t))...);
        }
    }

    template <typename VecT>
    constexpr static auto pop_right(VecT&& t)
    {
        return detail::copy_impl(std::forward<VecT>(t),
                                 make_index_sequence<VecTraits<std::decay_t<VecT>>::size() - 1>{});
    }

    template <typename VecT>
    constexpr static auto pop_left(VecT&& t)
    {
        auto pop_front = [](auto front, auto... rest) { return make_vector(rest...); };
        return std::apply(pop_front, std::forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static auto get_first(VecT&& t)
    {
        return std::get<0>(std::forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static auto get_last(VecT&& t)
    {
        return std::get<VecTraits<std::decay_t<VecT>>::size() - 1u>(std::forward<VecT>(t));
    }

    template <typename VecT, std::size_t... Indices>
    constexpr static auto reverse_impl(VecT&& t, index_sequence<Indices...>)
    {
        return make_vector(std::get<sizeof...(Indices) - 1 - Indices>(std::forward<VecT>(t))...);
    }

    template <typename VecT>
    constexpr static auto reverse(VecT&& t)
    {
        return reverse_impl(std::forward<VecT>(t),
                            make_index_sequence<VecTraits<std::decay_t<VecT>>::size()>{});
    }

    template <typename T, std::size_t... Indices>
    constexpr static auto flatten_coord_right_impl(T&& coord, T&& dims, index_sequence<Indices...>)
    {
        auto flatten = [](auto&& c, auto&& d, auto& mul) {
            auto result = c * mul;
            mul *= d;
            return result;
        };

        auto mult = 1;
        return (flatten(std::get<Indices>(coord), std::get<Indices>(dims), mult) + ...);
    }

    template <typename Lhs, typename Rhs>
    constexpr static auto flatten_coord_right(Lhs&& coord, Rhs&& dims)
    {
        return flatten_coord_right_impl(
            std::forward<Lhs>(coord),
            std::forward<Rhs>(dims),
            make_index_sequence<VecTraits<std::decay_t<Lhs>>::size()>{});
    }

    template <typename Lhs, typename Rhs, std::size_t... Indices>
    constexpr static auto
        flatten_coord_left_impl(Lhs&& coord, Rhs&& dims, index_sequence<Indices...>)
    {
        auto flatten = [](auto&& c, auto&& d, auto& mul) {
            auto result = c * mul;
            mul *= d;
            return result;
        };
        auto mult = 1;
        return (flatten(std::get<sizeof...(Indices) - 1 - Indices>(std::forward<Lhs>(coord)),
                        std::get<sizeof...(Indices) - 1 - Indices>(std::forward<Rhs>(dims)),
                        std::forward<decltype(mult)&>(mult))
                + ...);
    }

    template <typename Lhs, typename Rhs>
    constexpr static auto flatten_coord_left(Lhs&& coord, Rhs&& dims)
    {
        return flatten_coord_left_impl(std::forward<Lhs>(coord),
                                       std::forward<Rhs>(dims),
                                       make_index_sequence<VecTraits<std::decay_t<Lhs>>::size()>{});
    }

    template <typename Coord1d, typename T, std::size_t... Indices>
    constexpr static inline auto
        inflate_coord_right_impl(Coord1d const& flatCoord, T&& dims, index_sequence<Indices...>)
    {
        auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
            auto result = (last ? (c / div) : (c / div % d));
            div *= d;
            return result;
        };

        auto div = 1;
        return std::decay_t<T>{inflate(std::forward<Coord1d const&>(flatCoord),
                                       std::get<Indices>(std::forward<T>(dims)),
                                       std::forward<decltype(div)&>(div),
                                       Indices == sizeof...(Indices) - 1)...};
    }

    template <typename Coord1d, typename T>
    constexpr static inline auto inflate_coord_right(Coord1d const& flatCoord, T&& dims)
    {
        auto result
            = inflate_coord_right_impl(std::forward<decltype(flatCoord)>(flatCoord),
                                       std::forward<T>(dims),
                                       make_index_sequence<VecTraits<std::decay_t<T>>::size()>{});
        return result;
    }

    template <typename Coord1d, typename T, std::size_t... Indices>
    constexpr static inline auto
        inflate_coord_left_impl(Coord1d const& flatCoord, T&& dims, index_sequence<Indices...>)
    {
        auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
            auto result = (last ? (c / div) : (c / div % d));
            div *= d;
            return result;
        };

        auto div = 1;
        return reverse(std::decay_t<T>{
            inflate(flatCoord,
                    std::get<VecTraits<std::decay_t<T>>::size() - 1 - Indices>(dims),
                    div,
                    Indices == sizeof...(Indices) - 1)...});
    }

    template <typename Coord1d, typename T>
    constexpr static inline auto inflate_coord_left(Coord1d const& flatCoord, T&& dims)
    {
        auto result = inflate_coord_left_impl(
            flatCoord, dims, make_index_sequence<VecTraits<std::decay_t<T>>::size()>{});
        return result;
    }

    template <typename T, typename Y, std::size_t... Indices>
    constexpr static inline auto
        to_matrix_space_impl(T&& strides, Y&& strideCounts, index_sequence<Indices...>)
    {
        auto inflate = [](auto&& stride, auto&& count) { return count * stride; };

        return typename VecTraits<std::decay_t<T>>::DataT{
            (inflate(std::get<Indices>(strides), std::get<Indices>(strideCounts)) + ...)};
    }

    template <typename T, typename Y>
    constexpr static inline auto to_matrix_space(T&& strides, Y&& strideCounts)
    {
        auto result = to_matrix_space_impl(
            strides, strideCounts, make_index_sequence<VecTraits<std::decay_t<T>>::size()>{});
        return result;
    }

    template <class T, size_t... I>
    auto& print(std::ostream& os, T&& t, std::index_sequence<I...>&&)
    {
        os << "(";
        (..., (os << (I == 0 ? "" : ", ") << std::get<I>(std::forward<T>(t))));
        return os << ")\n";
    }

    template <class... ArgsT>
    auto& print(std::ostream& os, std::tuple<ArgsT...> const& t)
    {
        return print(os, t, std::make_index_sequence<sizeof...(ArgsT)>());
    }

} // namespace rocwmma

namespace std
{
    template <typename... Args>
    ostream& operator<<(ostream& os, tuple<Args...> const& t)
    {
        return rocwmma::print(os, t);
    }
}

#endif // ROCWMMA_TUPLE_HPP
