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
#ifndef ROCWMMA_TUPLE_HPP
#define ROCWMMA_TUPLE_HPP

#include <iostream>
#include <tuple>

namespace rocwmma
{
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator*(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} *= non_native_vector_base<T, n>{y};
    }

    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator*(U y, const non_native_vector_base<T, n>& x) noexcept
    {
        return non_native_vector_base<T, n>{x} *= non_native_vector_base<T, n>{y};
    }
}

namespace std
{

    template <typename T>
    struct is_tuple
    {
        constexpr static auto value = false;
    };

    template <typename... ArgsT>
    struct is_tuple<tuple<ArgsT...>>
    {
        constexpr static auto value = true;
    };

    template <typename T,
              typename... Types,
              size_t... Indices,
              typename std::enable_if_t<is_tuple<decay_t<T>>::value == false, int> = 0>
    constexpr static inline auto
        operator_mult_impl(T&& val, tuple<Types...> const& tup, index_sequence<Indices...>)
    {
        return make_tuple(val * get<Indices>(tup)...);
    }

    template <typename... TypesL, typename... TypesR, size_t... Indices>
    constexpr static inline auto operator_mult_impl(tuple<TypesL...> const& lhs,
                                                    tuple<TypesR...> const& rhs,
                                                    index_sequence<Indices...>)
    {
        return make_tuple(get<Indices>(lhs) * get<Indices>(rhs)...);
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator*(T&& val, tuple<Types...> const& tup)
    {
        return operator_mult_impl(std::forward<T>(val),
                                  std::forward<decltype(tup)>(tup),
                                  make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    // template <typename... TypesL, typename... TypesR>
    // constexpr static inline auto operator*(tuple<TypesL...> const& lhs, tuple<TypesR...> const& rhs)
    // {
    //     return operator_mult_impl(std::forward<decltype(lhs)>(lhs),
    //                               std::forward<decltype(rhs)>(rhs),
    //                               make_index_sequence<tuple_size<tuple<TypesL...>>::value>());
    // }

    template <typename T,
              typename... Types,
              size_t... Indices,
              typename std::enable_if_t<is_tuple<decay_t<T>>::value == false, int> = 0>
    constexpr static inline auto
        operator_add_impl(T&& lhs, tuple<Types...> const& rhs, index_sequence<Indices...>)
    {
        return make_tuple(lhs + get<Indices>(rhs)...);
    }

    template <typename T,
              typename... Types,
              size_t... Indices,
              typename std::enable_if_t<is_tuple<decay_t<T>>::value == false, int> = 0>
    constexpr static inline auto
        operator_add_impl(tuple<Types...> const& lhs, T&& rhs, index_sequence<Indices...>)
    {
        return make_tuple(get<Indices>(lhs) + rhs...);
    }

    template <typename... Types, size_t... Indices>
    constexpr static inline auto operator_add_impl(tuple<Types...> const& lhs,
                                                   tuple<Types...> const& rhs,
                                                   index_sequence<Indices...>)
    {
        return make_tuple(get<Indices>(lhs) + get<Indices>(rhs)...);
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator+(T&& lhs, tuple<Types...> const& rhs)
    {
        return operator_add_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator+(tuple<Types...> const& lhs, T&& rhs)
    {
        return operator_add_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename T,
              typename... Types,
              size_t... Indices,
              typename std::enable_if_t<is_tuple<decay_t<T>>::value == false, int> = 0>
    constexpr static inline auto
        operator_sub_impl(T&& lhs, tuple<Types...> const& rhs, index_sequence<Indices...>)
    {
        return make_tuple(lhs - get<Indices>(rhs)...);
    }

    template <typename T,
              typename... Types,
              size_t... Indices,
              typename std::enable_if_t<is_tuple<decay_t<T>>::value == false, int> = 0>
    constexpr static inline auto
        operator_sub_impl(tuple<Types...> const& lhs, T&& rhs, index_sequence<Indices...>)
    {
        return make_tuple(get<Indices>(lhs) - rhs...);
    }

    template <typename... Types, size_t... Indices>
    constexpr static inline auto operator_sub_impl(tuple<Types...> const& lhs,
                                                   tuple<Types...> const& rhs,
                                                   index_sequence<Indices...>)
    {
        return make_tuple(get<Indices>(lhs) - get<Indices>(rhs)...);
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator-(T&& lhs, tuple<Types...> const& rhs)
    {
        return operator_sub_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator-(tuple<Types...> const& lhs, T&& rhs)
    {
        return operator_sub_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

} // namespace std

namespace rocwmma
{

    // struct MakeTuple
    // {
    //     template<typename... ArgsT>
    //     auto operator()(ArgsT&&... args) { return std::make_tuple(std::forward<ArgsT>(args)...); }
    // };

    template <typename T, std::size_t... Indices>
    constexpr static auto copy_impl(T&& t, std::index_sequence<Indices...>&&)
    {
        return std::make_tuple(std::get<Indices>(std::forward<T>(t))...);
    }

    template <typename T>
    constexpr static auto pop_right(T&& t)
    {
        return copy_impl(std::forward<T>(t),
                         std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value - 1>{});
    }

    template <typename T>
    constexpr static auto pop_left(T&& t)
    {
        auto pop_front = [](auto front, auto... rest) { return std::make_tuple(rest...); };
        return std::apply(pop_front, std::forward<T>(t));
    }

    template <typename T, std::size_t... Indices>
    constexpr static auto reverse_impl(T&& t, std::index_sequence<Indices...>)
    {
        return std::make_tuple(std::get<sizeof...(Indices) - 1 - Indices>(std::forward<T>(t))...);
    }

    template <typename T>
    constexpr static auto reverse(T&& t)
    {
        return reverse_impl(std::forward<T>(t),
                            std::make_index_sequence<std::tuple_size<T>::value>());
    }

    template <typename T, std::size_t... Indices>
    constexpr static auto
        flatten_coord_right_impl(T&& coord, T&& dims, std::index_sequence<Indices...>)
    {
        auto flatten = [](auto&& c, auto&& d, auto& mul) {
            auto result = c * mul;
            mul *= d;
            return result;
        };

        auto mult = 1;
        return (flatten(std::get<Indices>(coord), std::get<Indices>(dims), mult) + ...);
    }

    template <typename T>
    constexpr static auto flatten_coord_right(T&& coord, T&& dims)
    {
        auto result = flatten_coord_right_impl(
            coord, dims, std::make_index_sequence<std::tuple_size<T>::value>());
        return result;
    }

    template <typename T, std::size_t... Indices>
    constexpr static auto
        flatten_coord_left_impl(T&& coord, T&& dims, std::index_sequence<Indices...>)
    {
        auto flatten = [](auto&& c, auto&& d, auto& mul) {
            auto result = c * mul;
            mul *= d;
            return result;
        };
        auto mult = 1;
        return (flatten(std::get<sizeof...(Indices) - 1 - Indices>(coord),
                        std::get<sizeof...(Indices) - 1 - Indices>(dims),
                        mult)
                + ...);
    }

    template <typename T>
    constexpr static auto flatten_coord_left(T&& coord, T&& dims)
    {
        auto result = flatten_coord_left_impl(
            coord, dims, std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
        return result;
    }

    template <typename Coord1d, typename T, std::size_t... Indices>
    constexpr static inline auto inflate_coord_right_impl(Coord1d const& flatCoord,
                                                          T&&            dims,
                                                          std::index_sequence<Indices...>)
    {
        auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
            auto result = (last ? (c / div) : (c / div % d));
            div *= d;
            return result;
        };

        auto div = 1;
        return std::decay_t<T>{
            inflate(flatCoord, std::get<Indices>(dims), div, Indices == sizeof...(Indices) - 1)...};
    }

    template <typename Coord1d, typename T>
    constexpr static inline auto inflate_coord_right(Coord1d const& flatCoord, T&& dims)
    {
        auto result = inflate_coord_right_impl(
            flatCoord, dims, std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
        return result;
    }

    template <typename Coord1d, typename T, std::size_t... Indices>
    constexpr static inline auto
        inflate_coord_left_impl(Coord1d const& flatCoord, T&& dims, std::index_sequence<Indices...>)
    {
        auto inflate = [](auto&& c, auto&& d, auto& div, bool last) {
            auto result = (last ? (c / div) : (c / div % d));
            div *= d;
            return result;
        };

        auto div = 1;
        return reverse(std::decay_t<T>{
            inflate(flatCoord,
                    std::get<std::tuple_size<std::decay_t<T>>::value - 1 - Indices>(dims),
                    div,
                    Indices == sizeof...(Indices) - 1)...});
    }

    template <typename Coord1d, typename T>
    constexpr static inline auto inflate_coord_left(Coord1d const& flatCoord, T&& dims)
    {
        auto result = inflate_coord_left_impl(
            flatCoord, dims, std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
        return result;
    }

    template <typename T, typename Y, std::size_t... Indices>
    constexpr static inline auto
        to_matrix_space_impl(T&& strides, Y&& strideCounts, std::index_sequence<Indices...>)
    {
        auto inflate = [](auto&& stride, auto&& count) { return count * stride; };

        return std::tuple_element_t<0, std::decay_t<T>>{
            (inflate(std::get<Indices>(strides), std::get<Indices>(strideCounts)) + ...)};
    }

    template <typename T, typename Y>
    constexpr static inline auto to_matrix_space(T&& strides, Y&& strideCounts)
    {
        auto result = to_matrix_space_impl(
            strides,
            strideCounts,
            std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
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
