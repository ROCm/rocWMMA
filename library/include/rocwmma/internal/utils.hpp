/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_UTILS_HPP
#define ROCWMMA_UTILS_HPP

#include <tuple>
#include <utility>

///////////////////////////////////////////////////////////
////////  std::apply fold expressions (<= C++14)  /////////
///////////////////////////////////////////////////////////
namespace std
{

#if !(__cplusplus >= 201703L)
    template <typename F, typename Tuple, size_t... I>
    auto apply_impl(F fn, Tuple t, std::index_sequence<I...>)
    {
        return fn(std::get<I>(t)...);
    }
    template <typename F, typename Tuple>
    auto apply(F fn, Tuple t)
    {
        const std::size_t size = std::tuple_size<Tuple>::value;
        return apply_impl(fn, t, std::make_index_sequence<size>());
    }
#endif

} // namespace std

///////////////////////////////////////////////////////////
/////////////  std::pair<T, T> extensions  ////////////////
///////////////////////////////////////////////////////////
namespace std
{
    // Single operand for swap
    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T> swap(pair<T, T> const& p)
    {
        return std::make_pair(std::get<1>(p), std::get<0>(p));
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T>& swap(pair<T, T>& p)
    {
        std::swap(std::get<0>(p), std::get<1>(p));
        return p;
    }

    // Add, sub operators
    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T> operator+(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T>& operator+=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) += get<0>(rhs);
        get<1>(lhs) += get<1>(rhs);
        return lhs;
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T> operator*(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) * get<0>(rhs), get<1>(lhs) * get<1>(rhs));
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T>& operator*=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) *= get<0>(rhs);
        get<1>(lhs) *= get<1>(rhs);
        return lhs;
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T> operator-(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) - get<0>(rhs), get<1>(lhs) - get<1>(rhs));
    }

    template <typename T>
    __host__ __device__ constexpr static inline pair<T, T>& operator-=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) -= get<0>(rhs);
        get<1>(lhs) -= get<1>(rhs);
        return lhs;
    }

} // namespace std

namespace rocwmma
{

    // Computes ceil(numerator/divisor) for integer types.
    template <typename intT1,
              class = typename std::enable_if<std::is_integral<intT1>::value>::type,
              typename intT2,
              class = typename std::enable_if<std::is_integral<intT2>::value>::type>
    static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
    {
        return (numerator + divisor - 1) / divisor;
    }

    // Calculate integer Log base 2
    template <uint32_t x>
    struct Log2
    {
        static constexpr uint32_t value = 1 + Log2<(x >> 1)>::value;
        static_assert(x % 2 == 0, "Integer input must be a power of 2");
    };

    template <>
    struct Log2<1>
    {
        static constexpr uint32_t value = 0;
    };

    template <>
    struct Log2<0>
    {
        static constexpr uint32_t value = 0;
    };

    // Create a bitmask of size BitCount, starting from the LSB bit
    template <uint32_t BitCount>
    struct LsbMask;

    template <>
    struct LsbMask<1>
    {
        enum : uint32_t
        {
            value = 0x1
        };
    };

    template <>
    struct LsbMask<0>
    {
        enum : uint32_t
        {
            value = 0x0
        };
    };

    template <uint32_t BitCount>
    struct LsbMask
    {
        enum : uint32_t
        {
            value = LsbMask<1>::value << (BitCount - 1) | LsbMask<BitCount - 1>::value
        };
    };

} // namespace rocwmma

#endif // ROCWMMA_UTILS_HPP
