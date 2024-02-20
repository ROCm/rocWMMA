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

#if !defined(__HIPCC_RTC__)

#include <iostream>
#include <tuple>

#endif // !defined(__HIPCC_RTC__)

#include "utility/forward.hpp"
#include "utility/sequence.hpp"
#include "utils.hpp"

namespace rocwmma
{
    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator+(non_native_vector_base<VecT, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} += non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator+(U x, non_native_vector_base<VecT, Rank> const& y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} += non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator-(non_native_vector_base<VecT, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} -= non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator-(U x, const non_native_vector_base<VecT, Rank>& y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} -= non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator*(const non_native_vector_base<VecT, Rank>& x, U y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} *= non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator*(U x, const non_native_vector_base<VecT, Rank>& y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} *= non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator/(const non_native_vector_base<VecT, Rank>& x, U y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} /= non_native_vector_base<VecT, Rank>{y};
    }

    template <typename VecT, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<VecT, Rank>
        operator/(U x, const non_native_vector_base<VecT, Rank>& y) noexcept
    {
        return non_native_vector_base<VecT, Rank>{x} /= non_native_vector_base<VecT, Rank>{y};
    }

    namespace detail
    {
        template <typename VecT, size_t... Indices>
        constexpr static auto copy_impl(VecT&& t, index_sequence<Indices...>&&)
        {
            return make_vector(get<Indices>(forward<VecT>(t))...);
        }
    }

    template <typename VecT>
    constexpr static auto pop_right(VecT&& t)
    {
        return detail::copy_impl(forward<VecT>(t),
                                 make_index_sequence<VecTraits<decay_t<VecT>>::size() - 1>{});
    }

    template <typename VecT>
    constexpr static auto pop_left(VecT&& t)
    {
        auto pop_front = [](auto front, auto... rest) { return make_vector(rest...); };
        return apply(pop_front, forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static decltype(auto) get_first(VecT&& t)
    {
        return get<0>(forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static decltype(auto) get_last(VecT&& t)
    {
        return get<VecTraits<decay_t<VecT>>::size() - 1u>(forward<VecT>(t));
    }

    namespace detail
    {
        template <typename VecT, size_t... Indices>
        constexpr static decltype(auto) reverse_impl(VecT&& t, index_sequence<Indices...>)
        {
            return make_vector(get<sizeof...(Indices) - 1 - Indices>(forward<VecT>(t))...);
        }
    }

    template <typename VecT>
    constexpr static decltype(auto) reverse(VecT&& t)
    {
        return detail::reverse_impl(forward<VecT>(t),
                                    make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_right_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<decay_t<Vec0>>::DataT{1};
            return (flatten(get<Indices>(forward<Vec0>(coord)),
                            get<Indices>(forward<Vec1>(dims)),
                            forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_right(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_right_impl(
            forward<Vec0>(coord),
            forward<Vec1>(dims),
            make_index_sequence<VecTraits<decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_left_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<decay_t<Vec0>>::DataT{1};
            return (flatten(get<sizeof...(Indices) - 1 - Indices>(forward<Vec0>(coord)),
                            get<sizeof...(Indices) - 1 - Indices>(forward<Vec1>(dims)),
                            forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_left(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_left_impl(
            forward<Vec0>(coord),
            forward<Vec1>(dims),
            make_index_sequence<VecTraits<decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_right_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
                auto result = (last ? (c / div) : (c / div % d));
                div *= d;
                return result;
            };

            auto div = decay_t<Coord1d>{1};
            return make_vector(inflate(forward<Coord1d>(flatCoord),
                                       get<Indices>(forward<VecT>(dims)),
                                       forward<decltype(div)&>(div),
                                       Indices == sizeof...(Indices) - 1)...);
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_right(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_right_impl(
            forward<Coord1d>(flatCoord),
            forward<VecT>(dims),
            make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_left_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
                auto result = (last ? (c / div) : (c / div % d));
                div *= d;
                return result;
            };

            auto div = decay_t<Coord1d>{1};
            return reverse(make_vector(
                inflate(forward<Coord1d>(flatCoord),
                        get<VecTraits<decay_t<VecT>>::size() - 1 - Indices>(forward<VecT>(dims)),
                        forward<decltype(div)&>(div),
                        Indices == sizeof...(Indices) - 1)...));
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_left(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_left_impl(
            forward<Coord1d>(flatCoord),
            forward<VecT>(dims),
            make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, size_t... Indices>
        constexpr static inline decltype(auto)
            to_matrix_space_impl(Vec0&& strides, Vec1&& strideSpace, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<Vec1>>::size() == sizeof...(Indices),
                          "strides and strideSpace vectors must be the same size");

            auto inflate = [](auto&& stride, auto&& dim) { return stride * dim; };

            return (inflate(get<Indices>(forward<Vec0>(strides)),
                            get<Indices>(forward<Vec1>(strideSpace)))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static inline decltype(auto) to_matrix_space(Vec0&& strides, Vec1&& strideSpace)
    {
        return detail::to_matrix_space_impl(
            forward<Vec0>(strides),
            forward<Vec1>(strideSpace),
            make_index_sequence<VecTraits<decay_t<Vec0>>::size()>{});
    }

#if !defined(__HIPCC_RTC__)

    template <class T, size_t... I>
    auto& print(std::ostream& os, T&& t, index_sequence<I...>&&)
    {
        os << "(";
        (..., (os << (I == 0 ? "" : ", ") << get<I>(forward<T>(t))));
        return os << ")\n";
    }

    template <class... ArgsT>
    auto& print(std::ostream& os, std::tuple<ArgsT...> const& t)
    {
        return print(os, t, make_index_sequence<sizeof...(ArgsT)>());
    }

#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{
    template <typename... Args>
    ostream& operator<<(ostream& os, tuple<Args...> const& t)
    {
        return rocwmma::print(os, t);
    }
}
#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_TUPLE_HPP
