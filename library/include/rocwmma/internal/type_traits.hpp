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
    ROCWMMA_HOST_DEVICE constexpr inline const T& max(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr inline const T& min(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }

    template <typename _Tp>
    struct __success_type
    {
        typedef _Tp type;
    };

    struct __failure_type
    {
    };

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

    template <typename>
    struct __is_member_object_pointer_helper : public false_type
    {
    };

    template <typename _Tp, typename _Cp>
    struct __is_member_object_pointer_helper<_Tp _Cp::*> : public __not_<is_function<_Tp>>::type
    {
    };

    /// is_member_object_pointer
    template <typename _Tp>
    struct is_member_object_pointer
        : public __is_member_object_pointer_helper<typename remove_cv<_Tp>::type>::type
    {
    };

    template <typename>
    struct __is_member_function_pointer_helper : public false_type
    {
    };

    template <typename _Tp, typename _Cp>
    struct __is_member_function_pointer_helper<_Tp _Cp::*> : public is_function<_Tp>::type
    {
    };

    /// is_member_function_pointer
    template <typename _Tp>
    struct is_member_function_pointer
        : public __is_member_function_pointer_helper<typename remove_cv<_Tp>::type>::type
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

    template <typename _Tp>
    class reference_wrapper;

    // Helper which adds a reference to a type when given a reference_wrapper
    template <typename _Tp>
    struct __strip_reference_wrapper
    {
        typedef _Tp __type;
    };

    template <typename _Tp>
    struct __strip_reference_wrapper<reference_wrapper<_Tp>>
    {
        typedef _Tp& __type;
    };

    template <typename _Tp>
    struct __decay_and_strip
    {
        typedef typename __strip_reference_wrapper<typename decay<_Tp>::type>::__type __type;
    };

    /// is_empty
    template <typename _Tp>
    struct is_empty : public integral_constant<bool, __is_empty(_Tp)>
    {
    };

    // __void_t (std::void_t for C++11)
    template <typename...>
    using __void_t = void;

    template <typename _Tp, typename _Up = _Tp&&>
    _Up __declval(int);

    template <typename _Tp>
    _Tp __declval(long);

    template <typename _Tp>
    auto declval() noexcept -> decltype(__declval<_Tp>(0));

    template <typename, unsigned = 0>
    struct extent;

    template <typename>
    struct remove_all_extents;

    /// is_constructible
    template <typename _Tp, typename... _Args>
    struct is_constructible : public __bool_constant<__is_constructible(_Tp, _Args...)>
    {
    };

    /// is_default_constructible
    template <typename _Tp>
    struct is_default_constructible : public is_constructible<_Tp>::type
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_copy_constructible_impl;

    template <typename _Tp>
    struct __is_copy_constructible_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_copy_constructible_impl<_Tp, true> : public is_constructible<_Tp, const _Tp&>
    {
    };

    /// is_copy_constructible
    template <typename _Tp>
    struct is_copy_constructible : public __is_copy_constructible_impl<_Tp>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_move_constructible_impl;

    template <typename _Tp>
    struct __is_move_constructible_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_move_constructible_impl<_Tp, true> : public is_constructible<_Tp, _Tp&&>
    {
    };

    /// is_move_constructible
    template <typename _Tp>
    struct is_move_constructible : public __is_move_constructible_impl<_Tp>
    {
    };

    template <bool, typename _Tp, typename... _Args>
    struct __is_nt_constructible_impl : public false_type
    {
    };

    template <typename _Tp, typename... _Args>
    struct __is_nt_constructible_impl<true, _Tp, _Args...>
        : public __bool_constant<noexcept(_Tp(std::declval<_Args>()...))>
    {
    };

    template <typename _Tp, typename _Arg>
    struct __is_nt_constructible_impl<true, _Tp, _Arg>
        : public __bool_constant<noexcept(static_cast<_Tp>(std::declval<_Arg>()))>
    {
    };

    template <typename _Tp>
    struct __is_nt_constructible_impl<true, _Tp> : public __bool_constant<noexcept(_Tp())>
    {
    };

    template <typename _Tp, size_t _Num>
    struct __is_nt_constructible_impl<true, _Tp[_Num]>
        : public __bool_constant<noexcept(typename remove_all_extents<_Tp>::type())>
    {
    };

    template <typename _Tp, typename... _Args>
    using __is_nothrow_constructible_impl
        = __is_nt_constructible_impl<__is_constructible(_Tp, _Args...), _Tp, _Args...>;

    /// is_nothrow_constructible
    template <typename _Tp, typename... _Args>
    struct is_nothrow_constructible : public __is_nothrow_constructible_impl<_Tp, _Args...>::type
    {
    };

    /// is_nothrow_default_constructible
    template <typename _Tp>
    struct is_nothrow_default_constructible : public __is_nothrow_constructible_impl<_Tp>::type
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nothrow_copy_constructible_impl;

    template <typename _Tp>
    struct __is_nothrow_copy_constructible_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_nothrow_copy_constructible_impl<_Tp, true>
        : public is_nothrow_constructible<_Tp, const _Tp&>
    {
    };

    /// is_nothrow_copy_constructible
    template <typename _Tp>
    struct is_nothrow_copy_constructible : public __is_nothrow_copy_constructible_impl<_Tp>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nothrow_move_constructible_impl;

    template <typename _Tp>
    struct __is_nothrow_move_constructible_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_nothrow_move_constructible_impl<_Tp, true>
        : public is_nothrow_constructible<_Tp, _Tp&&>
    {
    };

    struct __do_is_implicitly_default_constructible_impl
    {
        template <typename _Tp>
        static void __helper(const _Tp&);

        template <typename _Tp>
        static true_type __test(const _Tp&, decltype(__helper<const _Tp&>({}))* = 0);

        static false_type __test(...);
    };

    template <typename _Tp>
    struct __is_implicitly_default_constructible_impl
        : public __do_is_implicitly_default_constructible_impl
    {
        typedef decltype(__test(declval<_Tp>())) type;
    };

    template <typename _Tp>
    struct __is_implicitly_default_constructible_safe
        : public __is_implicitly_default_constructible_impl<_Tp>::type
    {
    };

    template <typename _Tp>
    struct __is_implicitly_default_constructible
        : public __and_<is_default_constructible<_Tp>,
                        __is_implicitly_default_constructible_safe<_Tp>>
    {
    };

    /// is_assignable
    template <typename _Tp, typename _Up>
    struct is_assignable : public __bool_constant<__is_assignable(_Tp, _Up)>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_copy_assignable_impl;

    template <typename _Tp>
    struct __is_copy_assignable_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_copy_assignable_impl<_Tp, true> : public is_assignable<_Tp&, const _Tp&>
    {
    };

    /// is_copy_assignable
    template <typename _Tp>
    struct is_copy_assignable : public __is_copy_assignable_impl<_Tp>
    {
    };

    /// is_nothrow_move_constructible
    template <typename _Tp>
    struct is_nothrow_move_constructible : public __is_nothrow_move_constructible_impl<_Tp>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_move_assignable_impl;

    template <typename _Tp>
    struct __is_move_assignable_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_move_assignable_impl<_Tp, true> : public is_assignable<_Tp&, _Tp&&>
    {
    };

    /// is_move_assignable
    template <typename _Tp>
    struct is_move_assignable : public __is_move_assignable_impl<_Tp>
    {
    };

    template <typename _Tp, typename _Up>
    struct __is_nt_assignable_impl
        : public integral_constant<bool, noexcept(declval<_Tp>() = declval<_Up>())>
    {
    };

    /// is_nothrow_assignable
    template <typename _Tp, typename _Up>
    struct is_nothrow_assignable
        : public __and_<is_assignable<_Tp, _Up>, __is_nt_assignable_impl<_Tp, _Up>>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nt_copy_assignable_impl;

    template <typename _Tp>
    struct __is_nt_copy_assignable_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_nt_copy_assignable_impl<_Tp, true> : public is_nothrow_assignable<_Tp&, const _Tp&>
    {
    };

    /// is_nothrow_copy_assignable
    template <typename _Tp>
    struct is_nothrow_copy_assignable : public __is_nt_copy_assignable_impl<_Tp>
    {
    };

    template <typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nt_move_assignable_impl;

    template <typename _Tp>
    struct __is_nt_move_assignable_impl<_Tp, false> : public false_type
    {
    };

    template <typename _Tp>
    struct __is_nt_move_assignable_impl<_Tp, true> : public is_nothrow_assignable<_Tp&, _Tp&&>
    {
    };

    /// is_nothrow_move_assignable
    template <typename _Tp>
    struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl<_Tp>
    {
    };

    template <typename _Tp>
    using __remove_cvref_t = typename remove_cv<typename remove_reference<_Tp>::type>::type;

    template <typename _Tp>
    struct __is_swappable;

    template <typename _Tp>
    struct __is_nothrow_swappable;

    template <typename... _Elements>
    class tuple;

    /// tuple_size
    template <typename _Tp>
    struct tuple_size;

    /// tuple_element
    template <std::size_t _Int, typename _Tp>
    struct tuple_element;

    template <typename>
    struct __is_tuple_like_impl : false_type
    {
    };

    template <typename... _Tps>
    struct __is_tuple_like_impl<tuple<_Tps...>> : true_type
    {
    };

    // Internal type trait that allows us to sfinae-protect tuple_cat.
    template <typename _Tp>
    struct __is_tuple_like : public __is_tuple_like_impl<__remove_cvref_t<_Tp>>::type
    {
    };

    template <typename _Tp>
    inline typename enable_if<__and_<__not_<__is_tuple_like<_Tp>>,
                                     is_move_constructible<_Tp>,
                                     is_move_assignable<_Tp>>::value>::type
        swap(_Tp&, _Tp&) noexcept(
            __and_<is_nothrow_move_constructible<_Tp>, is_nothrow_move_assignable<_Tp>>::value);

    template <typename _Tp, size_t _Nm>
    inline typename enable_if<__is_swappable<_Tp>::value>::type
        swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(__is_nothrow_swappable<_Tp>::value);

    namespace __swappable_details
    {
        using std::swap;

        struct __do_is_swappable_impl
        {
            template <typename _Tp,
                      typename = decltype(swap(std::declval<_Tp&>(), std::declval<_Tp&>()))>
            static true_type __test(int);

            template <typename>
            static false_type __test(...);
        };

        struct __do_is_nothrow_swappable_impl
        {
            template <typename _Tp>
            static __bool_constant<noexcept(swap(std::declval<_Tp&>(), std::declval<_Tp&>()))>
                __test(int);

            template <typename>
            static false_type __test(...);
        };

    } // namespace __swappable_details

    template <typename _Tp>
    struct __is_swappable_impl : public __swappable_details::__do_is_swappable_impl
    {
        typedef decltype(__test<_Tp>(0)) type;
    };

    template <typename _Tp>
    struct __is_nothrow_swappable_impl : public __swappable_details::__do_is_nothrow_swappable_impl
    {
        typedef decltype(__test<_Tp>(0)) type;
    };

    template <typename _Tp>
    struct __is_swappable : public __is_swappable_impl<_Tp>::type
    {
    };

    template <typename _Tp>
    struct __is_nothrow_swappable : public __is_nothrow_swappable_impl<_Tp>::type
    {
    };

    template <bool _Cond, typename _Tp = void>
    using __enable_if_t = typename enable_if<_Cond, _Tp>::type;

    template <typename _Tp>
    using remove_reference_t = typename remove_reference<_Tp>::type;

    struct __nonesuch
    {
        __nonesuch()                      = delete;
        ~__nonesuch()                     = delete;
        __nonesuch(__nonesuch const&)     = delete;
        void operator=(__nonesuch const&) = delete;
    };
    struct __nonesuch_no_braces : std::__nonesuch
    {
        explicit __nonesuch_no_braces(const __nonesuch&) = delete;
    };

    // Stores a tuple of indices.  Used by tuple and pair, and by bind() to
    // extract the elements in a tuple.
    template <size_t... _Indexes>
    struct _Index_tuple
    {
    };

#ifdef __has_builtin
#if __has_builtin(__make_integer_seq)
#define _GLIBCXX_USE_MAKE_INTEGER_SEQ 1
#endif
#endif

    // Builds an _Index_tuple<0, 1, 2, ..., _Num-1>.
    template <size_t _Num>
    struct _Build_index_tuple
    {
#if _GLIBCXX_USE_MAKE_INTEGER_SEQ
        template <typename, size_t... _Indices>
        using _IdxTuple = _Index_tuple<_Indices...>;

        using __type = __make_integer_seq<_IdxTuple, size_t, _Num>;
#else
        using __type = _Index_tuple<__integer_pack(_Num)...>;
#endif
    };

    // template<typename _Base, typename _Derived>
    // struct __is_base_of_helper
    // {
    //     typedef typename remove_cv<_Base>::type    _NoCv_Base;
    //     typedef typename remove_cv<_Derived>::type _NoCv_Derived;
    //     static const bool __value = (is_same<_Base, _Derived>::value
    // 			    || (__is_base_of(_Base, _Derived)
    // 			        && !is_same<_NoCv_Base,
    // 			                    _NoCv_Derived>::value));
    // };

    // template<typename _Base, typename _Derived>
    // struct is_base_of
    // : public integral_constant<bool,
    // 		       __is_base_of_helper<_Base, _Derived>::__value>
    // { };

    /// is_base_of
    template <typename _Base, typename _Derived>
    struct is_base_of : public integral_constant<bool, __is_base_of(_Base, _Derived)>
    {
    };

    template <typename _Tp, typename _Up>
    constexpr bool is_same_v = is_same<_Tp, _Up>::value;

    /// result_of
    template <typename _Signature>
    class result_of;

    struct __invoke_memfun_ref
    {
    };
    struct __invoke_memfun_deref
    {
    };
    struct __invoke_memobj_ref
    {
    };
    struct __invoke_memobj_deref
    {
    };
    struct __invoke_other
    {
    };

    // Associate a tag type with a specialization of __success_type.
    template <typename _Tp, typename _Tag>
    struct __result_of_success : __success_type<_Tp>
    {
        using __invoke_type = _Tag;
    };

    struct __result_of_memfun_ref_impl
    {
        template <typename _Fp, typename _Tp1, typename... _Args>
        static __result_of_success<decltype((std::declval<_Tp1>()
                                             .*std::declval<_Fp>())(std::declval<_Args>()...)),
                                   __invoke_memfun_ref>
            _S_test(int);

        template <typename...>
        static __failure_type _S_test(...);
    };

    template <typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun_ref : private __result_of_memfun_ref_impl
    {
        typedef decltype(_S_test<_MemPtr, _Arg, _Args...>(0)) type;
    };

    struct __result_of_memfun_deref_impl
    {
        template <typename _Fp, typename _Tp1, typename... _Args>
        static __result_of_success<decltype(((*std::declval<_Tp1>())
                                             .*std::declval<_Fp>())(std::declval<_Args>()...)),
                                   __invoke_memfun_deref>
            _S_test(int);

        template <typename...>
        static __failure_type _S_test(...);
    };

    template <typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun_deref : private __result_of_memfun_deref_impl
    {
        typedef decltype(_S_test<_MemPtr, _Arg, _Args...>(0)) type;
    };

    struct __result_of_memobj_ref_impl
    {
        template <typename _Fp, typename _Tp1>
        static __result_of_success<decltype(std::declval<_Tp1>().*std::declval<_Fp>()),
                                   __invoke_memobj_ref>
            _S_test(int);

        template <typename, typename>
        static __failure_type _S_test(...);
    };

    template <typename _MemPtr, typename _Arg>
    struct __result_of_memobj_ref : private __result_of_memobj_ref_impl
    {
        typedef decltype(_S_test<_MemPtr, _Arg>(0)) type;
    };

    struct __result_of_memobj_deref_impl
    {
        template <typename _Fp, typename _Tp1>
        static __result_of_success<decltype((*std::declval<_Tp1>()).*std::declval<_Fp>()),
                                   __invoke_memobj_deref>
            _S_test(int);

        template <typename, typename>
        static __failure_type _S_test(...);
    };

    template <typename _MemPtr, typename _Arg>
    struct __result_of_memobj_deref : private __result_of_memobj_deref_impl
    {
        typedef decltype(_S_test<_MemPtr, _Arg>(0)) type;
    };

    template <typename _MemPtr, typename _Arg>
    struct __result_of_memobj;

    template <typename _Res, typename _Class, typename _Arg>
    struct __result_of_memobj<_Res _Class::*, _Arg>
    {
        typedef __remove_cvref_t<_Arg> _Argval;
        typedef _Res _Class::*_MemPtr;
        typedef typename conditional<
            __or_<is_same<_Argval, _Class>, is_base_of<_Class, _Argval>>::value,
            __result_of_memobj_ref<_MemPtr, _Arg>,
            __result_of_memobj_deref<_MemPtr, _Arg>>::type::type type;
    };

    template <typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun;

    template <typename _Res, typename _Class, typename _Arg, typename... _Args>
    struct __result_of_memfun<_Res _Class::*, _Arg, _Args...>
    {
        typedef typename remove_reference<_Arg>::type _Argval;
        typedef _Res _Class::*_MemPtr;
        typedef typename conditional<is_base_of<_Class, _Argval>::value,
                                     __result_of_memfun_ref<_MemPtr, _Arg, _Args...>,
                                     __result_of_memfun_deref<_MemPtr, _Arg, _Args...>>::type::type
            type;
    };

    // Used by result_of, invoke etc. to unwrap a reference_wrapper.
    template <typename _Tp, typename _Up = __remove_cvref_t<_Tp>>
    struct __inv_unwrap
    {
        using type = _Tp;
    };

    template <typename _Tp, typename _Up>
    struct __inv_unwrap<_Tp, reference_wrapper<_Up>>
    {
        using type = _Up&;
    };

    template <bool, bool, typename _Functor, typename... _ArgTypes>
    struct __result_of_impl
    {
        typedef __failure_type type;
    };

    template <typename _MemPtr, typename _Arg>
    struct __result_of_impl<true, false, _MemPtr, _Arg>
        : public __result_of_memobj<typename decay<_MemPtr>::type,
                                    typename __inv_unwrap<_Arg>::type>
    {
    };

    template <typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_impl<false, true, _MemPtr, _Arg, _Args...>
        : public __result_of_memfun<typename decay<_MemPtr>::type,
                                    typename __inv_unwrap<_Arg>::type,
                                    _Args...>
    {
    };

    struct __result_of_other_impl
    {
        template <typename _Fn, typename... _Args>
        static __result_of_success<decltype(std::declval<_Fn>()(std::declval<_Args>()...)),
                                   __invoke_other>
            _S_test(int);

        template <typename...>
        static __failure_type _S_test(...);
    };

    template <typename _Functor, typename... _ArgTypes>
    struct __result_of_impl<false, false, _Functor, _ArgTypes...> : private __result_of_other_impl
    {
        typedef decltype(_S_test<_Functor, _ArgTypes...>(0)) type;
    };

    // __invoke_result (std::invoke_result for C++11)
    template <typename _Functor, typename... _ArgTypes>
    struct __invoke_result
        : public __result_of_impl<
              is_member_object_pointer<typename remove_reference<_Functor>::type>::value,
              is_member_function_pointer<typename remove_reference<_Functor>::type>::value,
              _Functor,
              _ArgTypes...>::type
    {
    };

    template <typename _Functor, typename... _ArgTypes>
    struct result_of<_Functor(_ArgTypes...)> : public __invoke_result<_Functor, _ArgTypes...>
    {
    };

    // The primary template is used for invalid INVOKE expressions.
    template <typename _Result, typename _Ret, bool = is_void<_Ret>::value, typename = void>
    struct __is_invocable_impl : false_type
    {
    };

    // Used for valid INVOKE and INVOKE<void> expressions.
    template <typename _Result, typename _Ret>
    struct __is_invocable_impl<_Result,
                               _Ret,
                               /* is_void<_Ret> = */ true,
                               __void_t<typename _Result::type>> : true_type
    {
    };

    // Used for INVOKE<R> expressions to check the implicit conversion to R.
    template <typename _Result, typename _Ret>
    struct __is_invocable_impl<_Result,
                               _Ret,
                               /* is_void<_Ret> = */ false,
                               __void_t<typename _Result::type>>
    {
    private:
        // The type of the INVOKE expression.
        // Unlike declval, this doesn't add_rvalue_reference.
        static typename _Result::type _S_get();

        template <typename _Tp>
        static void _S_conv(_Tp);

        // This overload is viable if INVOKE(f, args...) can convert to _Tp.
        template <typename _Tp, typename = decltype(_S_conv<_Tp>(_S_get()))>
        static true_type _S_test(int);

        template <typename _Tp>
        static false_type _S_test(...);

    public:
        using type = decltype(_S_test<_Ret>(1));
    };

    template <typename _Fn, typename... _ArgTypes>
    struct __is_invocable : __is_invocable_impl<__invoke_result<_Fn, _ArgTypes...>, void>::type
    {
    };

    template <typename _Fn, typename _Tp, typename... _Args>
    constexpr bool __call_is_nt(__invoke_memfun_ref)
    {
        using _Up = typename __inv_unwrap<_Tp>::type;
        return noexcept((std::declval<_Up>().*std::declval<_Fn>())(std::declval<_Args>()...));
    }

    template <typename _Fn, typename _Tp, typename... _Args>
    constexpr bool __call_is_nt(__invoke_memfun_deref)
    {
        return noexcept(((*std::declval<_Tp>()).*std::declval<_Fn>())(std::declval<_Args>()...));
    }

    template <typename _Fn, typename _Tp>
    constexpr bool __call_is_nt(__invoke_memobj_ref)
    {
        using _Up = typename __inv_unwrap<_Tp>::type;
        return noexcept(std::declval<_Up>().*std::declval<_Fn>());
    }

    template <typename _Fn, typename _Tp>
    constexpr bool __call_is_nt(__invoke_memobj_deref)
    {
        return noexcept((*std::declval<_Tp>()).*std::declval<_Fn>());
    }

    template <typename _Fn, typename... _Args>
    constexpr bool __call_is_nt(__invoke_other)
    {
        return noexcept(std::declval<_Fn>()(std::declval<_Args>()...));
    }

    template <typename _Result, typename _Fn, typename... _Args>
    struct __call_is_nothrow
        : __bool_constant<std::__call_is_nt<_Fn, _Args...>(typename _Result::__invoke_type{})>
    {
    };

    template <typename _Fn, typename... _Args>
    using __call_is_nothrow_ = __call_is_nothrow<__invoke_result<_Fn, _Args...>, _Fn, _Args...>;

    template <typename _Fn, typename... _Args>
    struct __is_nothrow_invocable
        : __and_<__is_invocable<_Fn, _Args...>, __call_is_nothrow_<_Fn, _Args...>>::type
    {
    };

} // namespace std
#endif // defined(__HIPCC_RTC__)

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
