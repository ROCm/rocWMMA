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

#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{
    ///////////////////////////////////////////////////////////////////
    ///           HIP_vector_type<T, N> utility overrides           ///
    ///                                                             ///
    /// Note: HIP_vector_type<T, N> uses vector extensions.         ///
    /// Element-wise access of vectors in constexpr is forbidden.   ///
    ///////////////////////////////////////////////////////////////////
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    __host__ __device__ constexpr inline DataT& get(HIP_vector_type<DataT, VecSize>& v)
    {
        return reinterpret_cast<DataT*>(&v.data)[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    __host__ __device__ constexpr inline DataT get(HIP_vector_type<DataT, VecSize> const& v)
    {
        return v.data[Idx];
    }

    template <typename DataT>
    __host__ __device__ constexpr inline auto swap(HIP_vector_type<DataT, 2> const& v)
    {
        return HIP_vector_type<DataT, 2>{get<1>(v), get<0>(v)};
    }

    ///////////////////////////////////////////////////////////////////
    ///     non_native_vector_base<T, N> utility overrides          ///
    ///////////////////////////////////////////////////////////////////
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    __host__ __device__ constexpr static inline DataT&
        get(non_native_vector_base<DataT, VecSize>& v)
    {
        return v[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    __host__ __device__ constexpr static inline DataT
        get(non_native_vector_base<DataT, VecSize> const& v)
    {
        return v[Idx];
    }

    // Unary swap only considered in 2d vectors.
    template <typename DataT>
    __host__ __device__ constexpr static inline auto swap(non_native_vector_base<DataT, 2> const& v)
    {
        return non_native_vector_base<DataT, 2>{get<1>(v), get<0>(v)};
    }

    ///////////////////////////////////////////////////////////////////
    ///                 Coord2d utility overrides                   ///
    ///                                                             ///
    /// Note: Coord2d MUST be constexpr compatible                  ///
    ///////////////////////////////////////////////////////////////////
    __host__ __device__ constexpr static inline auto make_coord2d(Coord2dDataT x, Coord2dDataT y)
    {
        return Coord2d{x, y};
    }

    __host__ __device__ constexpr static inline auto swap(Coord2d const& p)
    {
        return Coord2d{get<1>(p), get<0>(p)};
    }

    __host__ __device__ constexpr static inline Coord2d operator*(Coord2d const& lhs,
                                                                  Coord2d const& rhs)
    {
        return make_coord2d(get<0>(lhs) * get<0>(rhs), get<1>(lhs) * get<1>(rhs));
    }

    __host__ __device__ constexpr static inline Coord2d operator+(Coord2d const& lhs,
                                                                  Coord2d const& rhs)
    {
        return make_coord2d(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }
} // namespace rocwmma

///////////////////////////////////////////////////////////
////////  std::apply fold expressions (<= C++14)  /////////
///////////////////////////////////////////////////////////
#if !defined(__HIPCC_RTC__)
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
#endif // !defined(__HIPCC_RTC__)

///////////////////////////////////////////////////////////
///////////  std::numeric_limits for hipRTC  //////////////
///////////////////////////////////////////////////////////
#if defined(__HIPCC_RTC__)
namespace std
{
    template <typename T>
    class numeric_limits
    {
    public:
        static constexpr T min() noexcept;
        static constexpr T lowest() noexcept;
        static constexpr T max() noexcept;
        static constexpr T epsilon() noexcept;
        static constexpr T round_error() noexcept;
        static constexpr T infinity() noexcept;
        static constexpr T quiet_NaN() noexcept;
        static constexpr T signaling_NaN() noexcept;
        static constexpr T denorm_min() noexcept;
    };

} // namespace std
#endif

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

    // Helper for string representations of types
    template <typename DataT>
    constexpr const char* dataTypeToString();

    ///////////////////////////////////////////////////////////
    ///////////  rocwmma::dataTypeToString overloads  /////////
    ///////////////////////////////////////////////////////////

    template <>
    constexpr const char* dataTypeToString<float16_t>()
    {
        return "f16";
    }

    template <>
    constexpr const char* dataTypeToString<hfloat16_t>()
    {
        return "h16";
    }

    template <>
    constexpr const char* dataTypeToString<bfloat16_t>()
    {
        return "bf16";
    }

    template <>
    constexpr const char* dataTypeToString<float32_t>()
    {
        return "f32";
    }

    template <>
    constexpr const char* dataTypeToString<float64_t>()
    {
        return "f64";
    }

    template <>
    constexpr const char* dataTypeToString<int8_t>()
    {
        return "i8";
    }

    template <>
    constexpr const char* dataTypeToString<uint8_t>()
    {
        return "u8";
    }

    template <>
    constexpr const char* dataTypeToString<int32_t>()
    {
        return "i32";
    }

    template <>
    constexpr const char* dataTypeToString<uint32_t>()
    {
        return "u32";
    }

    template <>
    constexpr const char* dataTypeToString<row_major>()
    {
        return "T";
    }

    template <>
    constexpr const char* dataTypeToString<col_major>()
    {
        return "N";
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILS_HPP
