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

#ifndef ROCWMMA_UTILITY_TYPE_TRAITS_IMPL_HPP
#define ROCWMMA_UTILITY_TYPE_TRAITS_IMPL_HPP

namespace rocwmma
{
    namespace detail
    {
        // TODO: Separate file?
        template <typename T>
        ROCWMMA_HOST_DEVICE constexpr const T& max(const T& a, const T& b)
        {
            return (a < b) ? b : a;
        }

        template <typename T>
        ROCWMMA_HOST_DEVICE constexpr const T& min(const T& a, const T& b)
        {
            return (a < b) ? a : b;
        }

        using ::size_t;

        template <class T, T Val>
        struct integral_constant
        {
            static constexpr const T value = Val;
            using value_type               = T;
            using type                     = integral_constant;
            constexpr operator value_type() const
            {
                return value;
            }
            constexpr value_type operator()() const
            {
                return value;
            }
        };

        template <class T, T Val>
        constexpr const T integral_constant<T, Val>::value;

        using true_type  = integral_constant<bool, true>;
        using false_type = integral_constant<bool, false>;

        template <bool B>
        using bool_constant = integral_constant<bool, B>;

        using true_type  = bool_constant<true>;
        using false_type = bool_constant<false>;

        template <bool B>
        struct true_or_false_type : public false_type
        {
        };
        template <>
        struct true_or_false_type<true> : public true_type
        {
        };

        // Static conditional
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

        // Logical ops
        template <typename... Bs>
        struct logical_or;

        template <>
        struct logical_or<> : public false_type
        {
        };

        template <typename T>
        struct logical_or<T> : public T
        {
        };

        template <typename B1, typename B2>
        struct logical_or<B1, B2> : public conditional_t<B1::value, B1, B2>
        {
        };

        template <typename B1, typename B2, typename B3, typename... Bs>
        struct logical_or<B1, B2, B3, Bs...>
            : public conditional_t<B1::value, B1, logical_or<B2, B3, Bs...>>
        {
        };

        template <typename... Bs>
        using logical_or_t = typename logical_or<Bs...>::type;

        template <typename...>
        struct logical_and;

        template <>
        struct logical_and<> : public true_type
        {
        };

        template <typename B1>
        struct logical_and<B1> : public B1
        {
        };

        template <typename B1, typename B2>
        struct logical_and<B1, B2> : public conditional_t<B1::value, B2, B1>
        {
        };

        template <typename B1, typename B2, typename B3, typename... Bs>
        struct logical_and<B1, B2, B3, Bs...>
            : public conditional_t<B1::value, logical_and<B2, B3, Bs...>, B1>
        {
        };

        template <typename... Bs>
        using logical_and_t = typename logical_and<Bs...>::type;

        template <typename B>
        struct logical_not : public bool_constant<!bool(B::value)>
        {
        };

        template <typename B>
        using logical_not_t = typename logical_not<B>::type;

        // remove_reference
        template <typename T>
        struct remove_reference
        {
            using type = T;
        };

        template <typename T>
        struct remove_reference<T&>
        {
            using type = T;
        };

        template <typename T>
        struct remove_reference<T&&>
        {
            using type = T;
        };

        template <typename T>
        using remove_reference_t = typename remove_reference<T>::type;

        // remove_const
        template <typename T>
        struct remove_const
        {
            using type = T;
        };

        template <typename T>
        struct remove_const<T const>
        {
            using type = T;
        };

        template <typename T>
        using remove_const_t = typename remove_const<T>::type;

        // remove_volatile
        template <typename T>
        struct remove_volatile
        {
            using type = T;
        };

        template <typename T>
        struct remove_volatile<T volatile>
        {
            using type = T;
        };

        template <typename T>
        using remove_volatile_t = typename remove_volatile<T>::type;

        // remove_cv
        template <typename T>
        struct remove_cv
        {
            using type = remove_const_t<remove_volatile_t<T>>;
        };

        template <typename T>
        using remove_cv_t = typename remove_cv<T>::type;

        // remove_extent
        template <typename T>
        struct remove_extent
        {
            using type = T;
        };

        template <typename T, size_t _Size>
        struct remove_extent<T[_Size]>
        {
            using type = T;
        };

        template <typename T>
        struct remove_extent<T[]>
        {
            using type = T;
        };

        template <typename T>
        using remove_extent_t = typename remove_extent<T>::type;

        // add_pointer
        template <typename T>
        struct is_referenceable;

        template <typename T>
        struct is_void;

        template <typename T, bool = logical_or<is_referenceable<T>, is_void<T>>::value>
        struct add_pointer_helper
        {
            using type = T;
        };

        template <typename T>
        struct add_pointer_helper<T, true>
        {
            using type = remove_reference_t<T>*;
        };

        template <typename T>
        struct add_pointer : public add_pointer_helper<T>
        {
        };

        template <typename T>
        using add_pointer_t = typename add_pointer<T>::type;

        // is_lvalue_reference
        template <typename>
        struct is_lvalue_reference : public false_type
        {
        };

        template <typename T>
        struct is_lvalue_reference<T&> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_lvalue_reference_v = is_lvalue_reference<T>::value;

        // is_rvalue_reference
        template <typename>
        struct is_rvalue_reference : public false_type
        {
        };

        template <typename T>
        struct is_rvalue_reference<T&&> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_rvalue_reference_v = is_rvalue_reference<T>::value;

        // is_void
        template <typename>
        struct is_void_helper : public false_type
        {
        };

        template <>
        struct is_void_helper<void> : public true_type
        {
        };

        template <typename T>
        struct is_void : public is_void_helper<remove_cv_t<T>>::type
        {
        };

        template <typename T>
        inline constexpr bool is_void_v = is_void<T>::value;

        // is_reference
        template <typename T>
        struct is_reference : public logical_or_t<is_lvalue_reference<T>, is_rvalue_reference<T>>
        {
        };

        template <typename T>
        inline constexpr bool is_reference_v = is_reference<T>::value;

        // is_function
        template <typename>
        struct is_function : public false_type
        {
        };

        template <typename T>
        inline constexpr bool is_function_v = is_function<T>::value;

        // is_object
        template <typename T>
        struct is_object
            : public logical_not_t<logical_or<is_function<T>, is_reference<T>, is_void<T>>>
        {
        };

        template <typename T>
        inline constexpr bool is_object_v = is_object<T>::value;

        // __is_referenceable
        template <typename T>
        struct is_referenceable : public logical_or_t<is_object<T>, is_reference<T>>
        {
        };

        template <typename T>
        inline constexpr bool is_referenceable_v = is_referenceable<T>::value;

        // is_array
        template <typename>
        struct is_array : public false_type
        {
        };

        template <typename T, size_t _Size>
        struct is_array<T[_Size]> : public true_type
        {
        };

        template <typename T>
        struct is_array<T[]> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_array_v = is_array<T>::value;

        // is_integral
        template <class T>
        struct is_integral : public false_type
        {
        };
        template <>
        struct is_integral<bool> : public true_type
        {
        };
        template <>
        struct is_integral<char> : public true_type
        {
        };
        template <>
        struct is_integral<signed char> : public true_type
        {
        };
        template <>
        struct is_integral<unsigned char> : public true_type
        {
        };
        template <>
        struct is_integral<wchar_t> : public true_type
        {
        };
        template <>
        struct is_integral<short> : public true_type
        {
        };
        template <>
        struct is_integral<unsigned short> : public true_type
        {
        };
        template <>
        struct is_integral<int> : public true_type
        {
        };
        template <>
        struct is_integral<unsigned int> : public true_type
        {
        };
        template <>
        struct is_integral<long> : public true_type
        {
        };
        template <>
        struct is_integral<unsigned long> : public true_type
        {
        };
        template <>
        struct is_integral<long long> : public true_type
        {
        };
        template <>
        struct is_integral<unsigned long long> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_integral_v = is_integral<T>::value;

        // is_arithmetic
        template <class T>
        struct is_arithmetic : public false_type
        {
        };
        template <>
        struct is_arithmetic<bool> : public true_type
        {
        };
        template <>
        struct is_arithmetic<char> : public true_type
        {
        };
        template <>
        struct is_arithmetic<signed char> : public true_type
        {
        };
        template <>
        struct is_arithmetic<unsigned char> : public true_type
        {
        };
        template <>
        struct is_arithmetic<wchar_t> : public true_type
        {
        };
        template <>
        struct is_arithmetic<short> : public true_type
        {
        };
        template <>
        struct is_arithmetic<unsigned short> : public true_type
        {
        };
        template <>
        struct is_arithmetic<int> : public true_type
        {
        };
        template <>
        struct is_arithmetic<unsigned int> : public true_type
        {
        };
        template <>
        struct is_arithmetic<long> : public true_type
        {
        };
        template <>
        struct is_arithmetic<unsigned long> : public true_type
        {
        };
        template <>
        struct is_arithmetic<long long> : public true_type
        {
        };
        template <>
        struct is_arithmetic<unsigned long long> : public true_type
        {
        };
        template <>
        struct is_arithmetic<float> : public true_type
        {
        };
        template <>
        struct is_arithmetic<double> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

        // is_floating_point
        template <typename T>
        struct is_floating_point : public false_type
        {
        };
        template <>
        struct is_floating_point<float> : public true_type
        {
        };
        template <>
        struct is_floating_point<double> : public true_type
        {
        };
        template <>
        struct is_floating_point<long double> : public true_type
        {
        };

        template <typename T>
        inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

        // is_signed
        template <typename T, bool = is_arithmetic<T>::value>
        struct is_signed : public false_type
        {
        };

        template <typename T>
        struct is_signed<T, true> : public true_or_false_type<T(-1) < T(0)>
        {
        };

        template <typename T>
        inline constexpr bool is_signed_v = is_signed<T>::value;

        // is_same
        template <typename T, typename U>
        struct is_same : public false_type
        {
        };
        template <typename T>
        struct is_same<T, T> : public true_type
        {
        };

        template <class T, class U>
        inline constexpr bool is_same_v = is_same<T, U>::value;

        // is_convertible
        template <class T1, class T2>
        struct is_convertible : public true_or_false_type<__is_convertible_to(T1, T2)>
        {
        };

        template <class T, class U>
        inline constexpr bool is_convertible_v = is_convertible<T, U>::value;

        // decay selectors
        template <typename Up,
                  bool IsArray    = is_array<Up>::value,
                  bool IsFunction = is_function<Up>::value>
        struct decay_selector;

        template <typename Up>
        struct decay_selector<Up, false, false>
        {
            using type = remove_cv_t<Up>;
        };

        template <typename Up>
        struct decay_selector<Up, true, false>
        {
            using type = remove_extent_t<Up>*;
        };

        template <typename Up>
        struct decay_selector<Up, false, true>
        {
            using type = add_pointer_t<Up>;
        };

        template <typename T>
        using decay_selector_t = typename decay_selector<T>::type;

        // decay
        template <typename T>
        class decay
        {
            using remove_type = remove_reference_t<T>;

        public:
            using type = decay_selector_t<remove_type>;
        };

        template <typename T>
        using decay_t = typename decay<T>::type;

        // SFINAE enable_if
        template <bool B, class T = void>
        struct enable_if
        {
        };
        template <class T>
        struct enable_if<true, T>
        {
            using type = T;
        };

        template <bool B, class T = void>
        using enable_if_t = typename enable_if<B, T>::type;

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_TYPE_TRAITS_IMPL_HPP
