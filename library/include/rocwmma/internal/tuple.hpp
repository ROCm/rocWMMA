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
        return apply(pop_front, std::forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static decltype(auto) get_first(VecT&& t)
    {
        return std::get<0>(std::forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static decltype(auto) get_last(VecT&& t)
    {
        return std::get<VecTraits<std::decay_t<VecT>>::size() - 1u>(std::forward<VecT>(t));
    }

    namespace detail
    {
        template <typename VecT, std::size_t... Indices>
        constexpr static decltype(auto) reverse_impl(VecT&& t, index_sequence<Indices...>)
        {
            return make_vector(
                std::get<sizeof...(Indices) - 1 - Indices>(std::forward<VecT>(t))...);
        }
    }

    template <typename VecT>
    constexpr static decltype(auto) reverse(VecT&& t)
    {
        return detail::reverse_impl(std::forward<VecT>(t),
                                    make_index_sequence<VecTraits<std::decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, std::size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_right_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<std::decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<std::decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<std::decay_t<Vec0>>::DataT{1};
            return (flatten(std::get<Indices>(std::forward<Vec0>(coord)),
                            std::get<Indices>(std::forward<Vec1>(dims)),
                            std::forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_right(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_right_impl(
            std::forward<Vec0>(coord),
            std::forward<Vec1>(dims),
            make_index_sequence<VecTraits<std::decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, std::size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_left_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<std::decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<std::decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<std::decay_t<Vec0>>::DataT{1};
            return (flatten(std::get<sizeof...(Indices) - 1 - Indices>(std::forward<Vec0>(coord)),
                            std::get<sizeof...(Indices) - 1 - Indices>(std::forward<Vec1>(dims)),
                            std::forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_left(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_left_impl(
            std::forward<Vec0>(coord),
            std::forward<Vec1>(dims),
            make_index_sequence<VecTraits<std::decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, std::size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_right_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
                auto result = (last ? (c / div) : (c / div % d));
                div *= d;
                return result;
            };

            auto div = std::decay_t<Coord1d>{1};
            return make_vector(inflate(std::forward<Coord1d>(flatCoord),
                                       std::get<Indices>(std::forward<VecT>(dims)),
                                       std::forward<decltype(div)&>(div),
                                       Indices == sizeof...(Indices) - 1)...);
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_right(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_right_impl(
            std::forward<Coord1d>(flatCoord),
            std::forward<VecT>(dims),
            make_index_sequence<VecTraits<std::decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, std::size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_left_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
                auto result = (last ? (c / div) : (c / div % d));
                div *= d;
                return result;
            };

            auto div = std::decay_t<Coord1d>{1};
            return reverse(
                make_vector(inflate(std::forward<Coord1d>(flatCoord),
                                    std::get<VecTraits<std::decay_t<VecT>>::size() - 1 - Indices>(
                                        std::forward<VecT>(dims)),
                                    std::forward<decltype(div)&>(div),
                                    Indices == sizeof...(Indices) - 1)...));
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_left(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_left_impl(
            std::forward<Coord1d>(flatCoord),
            std::forward<VecT>(dims),
            make_index_sequence<VecTraits<std::decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, std::size_t... Indices>
        constexpr static inline decltype(auto)
            to_matrix_space_impl(Vec0&& strides, Vec1&& strideSpace, index_sequence<Indices...>)
        {
            static_assert(VecTraits<std::decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<std::decay_t<Vec1>>::size() == sizeof...(Indices),
                          "strides and strideSpace vectors must be the same size");

            auto inflate = [](auto&& stride, auto&& dim) { return stride * dim; };

            return (inflate(std::get<Indices>(std::forward<Vec0>(strides)),
                            std::get<Indices>(std::forward<Vec1>(strideSpace)))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static inline decltype(auto) to_matrix_space(Vec0&& strides, Vec1&& strideSpace)
    {
        return detail::to_matrix_space_impl(
            std::forward<Vec0>(strides),
            std::forward<Vec1>(strideSpace),
            make_index_sequence<VecTraits<std::decay_t<Vec0>>::size()>{});
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
