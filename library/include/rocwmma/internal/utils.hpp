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
/////////////  std replacements for hipRTC  ///////////////
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

    template <bool B, class T = void>
    using enable_if_t = typename enable_if<B, T>::type;

    template <bool B, class T, class F>
    struct conditional
    {
    };

    template <class T, class F>
    struct conditional<true, T, F>
    {
        using type = T;
    };

    template <class T, class F>
    struct conditional<false, T, F>
    {
        using type = F;
    };

    template <bool B, class T, class F>
    using conditional_t = typename conditional<B, T, F>::type;

    template <typename T>
    __HOST_DEVICE__ const T& max(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }

    template <typename T>
    __HOST_DEVICE__ const T& min(const T& a, const T& b)
    {
        return (b < a) ? a : b;
    }

    // Meta programming helper types.

    template <bool, typename, typename>
    struct conditional;

    template <typename...>
    struct __or_;

    template <>
    struct __or_<> : public false_type
    {
    };

    template <typename _B1>
    struct __or_<_B1> : public _B1
    {
    };

    template <typename _B1, typename _B2>
    struct __or_<_B1, _B2> : public conditional<_B1::value, _B1, _B2>::type
    {
    };

    template <typename _B1, typename _B2, typename _B3, typename... _Bn>
    struct __or_<_B1, _B2, _B3, _Bn...>
        : public conditional<_B1::value, _B1, __or_<_B2, _B3, _Bn...>>::type
    {
    };

    template <typename...>
    struct __and_;

    template <>
    struct __and_<> : public true_type
    {
    };

    template <typename _B1>
    struct __and_<_B1> : public _B1
    {
    };

    template <typename _B1, typename _B2>
    struct __and_<_B1, _B2> : public conditional<_B1::value, _B2, _B1>::type
    {
    };

    template <typename _B1, typename _B2, typename _B3, typename... _Bn>
    struct __and_<_B1, _B2, _B3, _Bn...>
        : public conditional<_B1::value, __and_<_B2, _B3, _Bn...>, _B1>::type
    {
    };

    template <bool __v>
    using __bool_constant = integral_constant<bool, __v>;

    template <typename _Pp>
    struct __not_ : public __bool_constant<!bool(_Pp::value)>
    {
    };

    // remove_reference
    template <typename T>
    struct remove_reference
    {
        typedef T type;
    };

    template <typename T>
    struct remove_reference<T&>
    {
        typedef T type;
    };

    template <typename T>
    struct remove_reference<T&&>
    {
        typedef T type;
    };

    // is_lvalue_reference
    template <typename>
    struct is_lvalue_reference : public false_type
    {
    };

    template <typename T>
    struct is_lvalue_reference<T&> : public true_type
    {
    };

    // is_rvalue_reference
    template <typename>
    struct is_rvalue_reference : public false_type
    {
    };

    template <typename _Tp>
    struct is_rvalue_reference<_Tp&&> : public true_type
    {
    };

    // lvalue forwarding
    template <typename T>
    constexpr T&& forward(typename remove_reference<T>::type& __t) noexcept
    {
        return static_cast<T&&>(__t);
    }

    // rvalue forwarding
    template <typename T>
    constexpr T&& forward(typename remove_reference<T>::type&& __t) noexcept
    {
        static_assert(!is_lvalue_reference<T>::value,
                      "template argument"
                      " substituting T is an lvalue reference type");
        return static_cast<T&&>(__t);
    }

    // remove_const
    template <typename T>
    struct remove_const
    {
        typedef T type;
    };

    template <typename T>
    struct remove_const<T const>
    {
        typedef T type;
    };

    // remove_volatile
    template <typename T>
    struct remove_volatile
    {
        typedef T type;
    };

    template <typename T>
    struct remove_volatile<T volatile>
    {
        typedef T type;
    };

    // remove_cv
    template <typename T>
    struct remove_cv
    {
        typedef typename remove_const<typename remove_volatile<T>::type>::type type;
    };

    // remove_extent
    template <typename T>
    struct remove_extent
    {
        typedef T type;
    };

    template <typename T, std::size_t _Size>
    struct remove_extent<T[_Size]>
    {
        typedef T type;
    };

    template <typename T>
    struct remove_extent<T[]>
    {
        typedef T type;
    };

    // is_void
    template <typename>
    struct __is_void_helper : public false_type
    {
    };

    template <>
    struct __is_void_helper<void> : public true_type
    {
    };

    template <typename _Tp>
    struct is_void : public __is_void_helper<typename remove_cv<_Tp>::type>::type
    {
    };

    // is_reference
    template <typename _Tp>
    struct is_reference : public __or_<is_lvalue_reference<_Tp>, is_rvalue_reference<_Tp>>::type
    {
    };

    // is_function
    template <typename>
    struct is_function : public false_type
    {
    };

    // is_object
    template <typename _Tp>
    struct is_object : public __not_<__or_<is_function<_Tp>, is_reference<_Tp>, is_void<_Tp>>>::type
    {
    };

    // __is_referenceable
    template <typename _Tp>
    struct __is_referenceable : public __or_<is_object<_Tp>, is_reference<_Tp>>::type
    {
    };

    // add_pointer
    template <typename T, bool = __or_<__is_referenceable<T>, is_void<T>>::value>
    struct __add_pointer_helper
    {
        typedef T type;
    };

    template <typename T>
    struct __add_pointer_helper<T, true>
    {
        typedef typename remove_reference<T>::type* type;
    };

    template <typename T>
    struct add_pointer : public __add_pointer_helper<T>
    {
    };

    // is_array
    template <typename>
    struct is_array : public false_type
    {
    };

    template <typename T, std::size_t _Size>
    struct is_array<T[_Size]> : public true_type
    {
    };

    template <typename T>
    struct is_array<T[]> : public true_type
    {
    };

    // decay selectors
    template <typename _Up,
              bool _IsArray    = is_array<_Up>::value,
              bool _IsFunction = is_function<_Up>::value>
    struct __decay_selector;

    template <typename _Up>
    struct __decay_selector<_Up, false, false>
    {
        typedef typename remove_cv<_Up>::type __type;
    };

    template <typename _Up>
    struct __decay_selector<_Up, true, false>
    {
        typedef typename remove_extent<_Up>::type* __type;
    };

    template <typename _Up>
    struct __decay_selector<_Up, false, true>
    {
        typedef typename add_pointer<_Up>::type __type;
    };

    // decay
    template <typename T>
    class decay
    {
        typedef typename remove_reference<T>::type __remove_type;

    public:
        typedef typename __decay_selector<__remove_type>::__type type;
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
