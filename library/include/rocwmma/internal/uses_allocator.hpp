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
#ifndef ROCWMMA_USES_ALLOCATOR_HPP
#define ROCWMMA_USES_ALLOCATOR_HPP

#include "type_traits.hpp"
#include "move.hpp"

namespace std
{
    // This is used for std::experimental::erased_type from Library Fundamentals.
    struct __erased_type { };

    // This also supports the "type-erased allocator" protocol from the
    // Library Fundamentals TS, where allocator_type is erased_type.
    // The second condition will always be false for types not using the TS.
    template<typename _Alloc, typename _Tp>
    using __is_erased_or_convertible
        = __or_<is_convertible<_Alloc, _Tp>, is_same<_Tp, __erased_type>>;

    /// [allocator.tag]
    struct allocator_arg_t { explicit allocator_arg_t() = default; };

    inline constexpr allocator_arg_t allocator_arg =
    allocator_arg_t();

    template<typename _Tp, typename _Alloc, typename = __void_t<>>
    struct __uses_allocator_helper
        : false_type { };

    template<typename _Tp, typename _Alloc>
    struct __uses_allocator_helper<_Tp, _Alloc,
				   __void_t<typename _Tp::allocator_type>>
        : __is_erased_or_convertible<_Alloc, typename _Tp::allocator_type>::type
        { };

    /// [allocator.uses.trait]
    template<typename _Tp, typename _Alloc>
    struct uses_allocator
        : __uses_allocator_helper<_Tp, _Alloc>::type
        { };

    struct __uses_alloc_base { };

    struct __uses_alloc0 : __uses_alloc_base
    {
        struct _Sink { void operator=(const void*) { } } _M_a;
    };

    template<typename _Alloc>
    struct __uses_alloc1 : __uses_alloc_base { const _Alloc* _M_a; };

    template<typename _Alloc>
    struct __uses_alloc2 : __uses_alloc_base { const _Alloc* _M_a; };

    template<bool, typename _Tp, typename _Alloc, typename... _Args>
    struct __uses_alloc;

    template<typename _Tp, typename _Alloc, typename... _Args>
    struct __uses_alloc<true, _Tp, _Alloc, _Args...>
        : conditional<
            is_constructible<_Tp, allocator_arg_t, const _Alloc&, _Args...>::value,
            __uses_alloc1<_Alloc>,
       	    __uses_alloc2<_Alloc>>::type
    {
        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2586. Wrong value category used in scoped_allocator_adaptor::construct
        static_assert(__or_<
	        is_constructible<_Tp, allocator_arg_t, const _Alloc&, _Args...>,
	        is_constructible<_Tp, _Args..., const _Alloc&>>::value,
	        "construction with an allocator must be possible"
	        " if uses_allocator is true");
    };

    template<typename _Tp, typename _Alloc, typename... _Args>
    struct __uses_alloc<false, _Tp, _Alloc, _Args...>
        : __uses_alloc0 { };

    template<typename _Tp, typename _Alloc, typename... _Args>
    using __uses_alloc_t =
        __uses_alloc<uses_allocator<_Tp, _Alloc>::value, _Tp, _Alloc, _Args...>;

    template<typename _Tp, typename _Alloc, typename... _Args>
    inline __uses_alloc_t<_Tp, _Alloc, _Args...>
    __use_alloc(const _Alloc& __a)
    {
        __uses_alloc_t<_Tp, _Alloc, _Args...> __ret;
        __ret._M_a = std::__addressof(__a);
        return __ret;
    }

    template<typename _Tp, typename _Alloc, typename... _Args>
    void __use_alloc(const _Alloc&&) = delete;

#if __cplusplus > 201402L
    template <typename _Tp, typename _Alloc>
    inline constexpr bool uses_allocator_v =
        uses_allocator<_Tp, _Alloc>::value;
#endif // C++17

    template<template<typename...> class _Predicate,
        typename _Tp, typename _Alloc, typename... _Args>
    struct __is_uses_allocator_predicate
        : conditional<uses_allocator<_Tp, _Alloc>::value,
            __or_<_Predicate<_Tp, allocator_arg_t, _Alloc, _Args...>,
	        _Predicate<_Tp, _Args..., _Alloc>>,
            _Predicate<_Tp, _Args...>>::type { };

    template<typename _Tp, typename _Alloc, typename... _Args>
    struct __is_uses_allocator_constructible
        : __is_uses_allocator_predicate<is_constructible, _Tp, _Alloc, _Args...>
        { };

#if __cplusplus >= 201402L
    template<typename _Tp, typename _Alloc, typename... _Args>
    inline constexpr bool __is_uses_allocator_constructible_v =
        __is_uses_allocator_constructible<_Tp, _Alloc, _Args...>::value;
#endif // C++14

    template<typename _Tp, typename _Alloc, typename... _Args>
    struct __is_nothrow_uses_allocator_constructible
        : __is_uses_allocator_predicate<is_nothrow_constructible,
			_Tp, _Alloc, _Args...>
    { };


#if __cplusplus >= 201402L
    template<typename _Tp, typename _Alloc, typename... _Args>
    inline constexpr bool
    __is_nothrow_uses_allocator_constructible_v =
        __is_nothrow_uses_allocator_constructible<_Tp, _Alloc, _Args...>::value;
#endif // C++14

    template<typename _Tp, typename... _Args>
    void __uses_allocator_construct_impl(__uses_alloc0 __a, _Tp* __ptr,
					 _Args&&... __args)
    { ::new ((void*)__ptr) _Tp(std::forward<_Args>(__args)...); }

    template<typename _Tp, typename _Alloc, typename... _Args>
    void __uses_allocator_construct_impl(__uses_alloc1<_Alloc> __a, _Tp* __ptr,
					 _Args&&... __args)
    {
        ::new ((void*)__ptr) _Tp(allocator_arg, *__a._M_a,
			       std::forward<_Args>(__args)...);
    }

    template<typename _Tp, typename _Alloc, typename... _Args>
    void __uses_allocator_construct_impl(__uses_alloc2<_Alloc> __a, _Tp* __ptr,
					 _Args&&... __args)
    { ::new ((void*)__ptr) _Tp(std::forward<_Args>(__args)..., *__a._M_a); }

    template<typename _Tp, typename _Alloc, typename... _Args>
    void __uses_allocator_construct(const _Alloc& __a, _Tp* __ptr,
				    _Args&&... __args)
    {
        std::__uses_allocator_construct_impl(
	    std::__use_alloc<_Tp, _Alloc, _Args...>(__a), __ptr,
	    std::forward<_Args>(__args)...);
    }
}





#endif // ROCWMMA_USES_ALLOCATOR_HPP