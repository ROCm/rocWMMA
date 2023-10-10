/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_TYPE_TRAITS_HPP
#define ROCWMMA_TYPE_TRAITS_HPP

#if !defined(__HIPCC_RTC__)

#include <cfloat>

#else

#define FLT_EPSILON __FLT_EPSILON__
#define FLT_MAX __FLT_MAX__
#define FLT_MIN __FLT_MIN__
#define HUGE_VALF (__builtin_huge_valf())

#endif // !defined(__HIPCC_RTC__)

#include "types.hpp"

namespace rocwmma
{
    namespace detail
    {
        struct Fp8Bits
        {
            union
            {
                uint8_t   i8;
                float8_t  f8;
                bfloat8_t bf8;
            };
            constexpr Fp8Bits(uint8_t initVal)
                : i8(initVal)
            {
            }
            constexpr Fp8Bits(float8_t initVal)
                : f8(initVal)
            {
            }
            constexpr Fp8Bits(bfloat8_t initVal)
                : bf8(initVal)
            {
            }
        };

        struct Fp16Bits
        {
            union
            {
                uint16_t  i16;
                float16_t f16;
#if !ROCWMMA_NO_HALF
                hfloat16_t h16;
#endif // !ROCWMMA_NO_HALF
                bfloat16_t b16;
            };
            constexpr Fp16Bits(uint16_t initVal)
                : i16(initVal)
            {
            }
            constexpr Fp16Bits(float16_t initVal)
                : f16(initVal)
            {
            }
#if !ROCWMMA_NO_HALF
            constexpr Fp16Bits(hfloat16_t initVal)
                : h16(initVal)
            {
            }
#endif
            constexpr Fp16Bits(bfloat16_t initVal)
                : b16(initVal)
            {
            }
        };

        struct Fp32Bits
        {
            union
            {
                uint32_t   i32;
                float32_t  f32;
                xfloat32_t xf32;
            };
            constexpr Fp32Bits(uint32_t initVal)
                : i32(initVal)
            {
            }
            constexpr Fp32Bits(float32_t initVal)
                : f32(initVal)
            {
            }
            constexpr Fp32Bits(xfloat32_t initVal)
                : xf32(initVal)
            {
            }
        };

    } // namespace detail
} // namespace rocwmma

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
        ROCWMMA_HOST_DEVICE static constexpr T min() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T lowest() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T max() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T epsilon() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T round_error() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T infinity() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T quiet_NaN() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T signaling_NaN() noexcept;
        ROCWMMA_HOST_DEVICE static constexpr T denorm_min() noexcept;
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
    ROCWMMA_HOST_DEVICE const T& max(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE const T& min(const T& a, const T& b)
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

    template <typename T>
    struct is_rvalue_reference<T&&> : public true_type
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

    template <typename T>
    struct is_void : public __is_void_helper<typename remove_cv<T>::type>::type
    {
    };

    // is_reference
    template <typename T>
    struct is_reference : public __or_<is_lvalue_reference<T>, is_rvalue_reference<T>>::type
    {
    };

    // is_function
    template <typename>
    struct is_function : public false_type
    {
    };

    // is_object
    template <typename T>
    struct is_object : public __not_<__or_<is_function<T>, is_reference<T>, is_void<T>>>::type
    {
    };

    // __is_referenceable
    template <typename T>
    struct __is_referenceable : public __or_<is_object<T>, is_reference<T>>::type{};

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

    template <typename T>
    using decay_t = typename decay<T>::type;

} // namespace std
#endif

namespace std
{
#if defined(__HIPCC_RTC__)
    using uint16_t = rocwmma::uint16_t;
#endif

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<float8_t>  //////////////
    ///////////////////////////////////////////////////////////
    // @cond
    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x28));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::infinity() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::lowest() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0xFF));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::max() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x7F));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::min() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x01));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.f8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float8_t
        numeric_limits<rocwmma::float8_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.f8;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<bfloat8_t>  //////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x38));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::infinity() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::lowest() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0xFF));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::max() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x7F));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::min() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x01));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.bf8;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat8_t
        numeric_limits<rocwmma::bfloat8_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp8Bits eps(static_cast<uint8_t>(0x80));
        return eps.bf8;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<float16_t>  //////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.f16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<hfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////
#if !ROCWMMA_NO_HALF
    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.h16;
    }

#endif // !ROCWMMA_NO_HALF

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<bfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x3C00));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F80));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFF7F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F7F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x007F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<xfloat32_t>  //////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<float>(FLT_EPSILON));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::infinity() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<float>(HUGE_VALF));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::lowest() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<float>(-FLT_MAX));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::max() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<float>(FLT_MAX));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::min() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<float>(FLT_MIN));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<uint32_t>(0x7FF80000));
        return eps.xf32;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::xfloat32_t
        numeric_limits<rocwmma::xfloat32_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp32Bits eps(static_cast<uint32_t>(0x7FF00000));
        return eps.xf32;
    }
    // @endcond

} // namespace std

namespace rocwmma
{
#if !defined(__HIPCC_RTC__)
    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
    constexpr auto maxExactInteger() -> decltype(std::numeric_limits<T>::max())
    {
        return std::numeric_limits<T>::max();
    }

    template <typename T,
              typename std::enable_if_t<std::is_floating_point<T>::value
                                            && std::numeric_limits<T>::digits,
                                        int>
              = 0>
    constexpr auto maxExactInteger() ->
        typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>
    {
        using RetT =
            typename std::conditional_t<std::is_same<T, float64_t>::value, int64_t, int32_t>;
        return ((RetT)1 << std::numeric_limits<T>::digits);
    }

    template <typename T,
              typename std::enable_if_t<
#if !ROCWMMA_NO_HALF
                  std::is_same<T, hfloat16_t>::value ||
#endif // !ROCWMMA_NO_HALF
                      std::is_same<T, float16_t>::value,
                  int>
              = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f16 mantissa is 10 bits
        return ((int32_t)1 << 11);
    }

    template <typename T, typename std::enable_if_t<std::is_same<T, bfloat16_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // b16 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }

    template <typename T, typename std::enable_if_t<std::is_same<T, float8_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f8 mantissa is 3 bits
        return ((int32_t)1 << 4);
    }

    template <typename T, typename std::enable_if_t<std::is_same<T, bfloat8_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // bf8 mantissa is 2 bits
        return ((int32_t)1 << 3);
    }

    template <typename T, typename std::enable_if_t<std::is_same<T, xfloat32_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // xf32 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }
#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

#endif // ROCWMMA_TYPE_TRAITS_HPP
