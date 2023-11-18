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
#else
#include "uses_allocator.hpp"
#include "utils.hpp"
#include "vector.hpp"

namespace std
{
    template <typename... _Elements>
    class tuple;

    template <typename _Tp>
    struct __is_empty_non_tuple : is_empty<_Tp>
    {
    };

    // Using EBO for elements that are tuples causes ambiguous base errors.
    template <typename _El0, typename... _El>
    struct __is_empty_non_tuple<tuple<_El0, _El...>> : false_type
    {
    };

    // Use the Empty Base-class Optimization for empty, non-final types.
    template <typename _Tp>
    using __empty_not_final =
        typename conditional<__is_final(_Tp), false_type, __is_empty_non_tuple<_Tp>>::type;

    template <std::size_t _Idx, typename _Head, bool = __empty_not_final<_Head>::value>
    struct _Head_base;

    template <std::size_t _Idx, typename _Head>
    struct _Head_base<_Idx, _Head, true> : public _Head
    {
        constexpr _Head_base()
            : _Head()
        {
        }

        constexpr _Head_base(const _Head& __h)
            : _Head(__h)
        {
        }

        constexpr _Head_base(const _Head_base&) = default;
        constexpr _Head_base(_Head_base&&)      = default;

        template <typename _UHead>
        constexpr _Head_base(_UHead&& __h)
            : _Head(std::forward<_UHead>(__h))
        {
        }

        _Head_base(allocator_arg_t, __uses_alloc0)
            : _Head()
        {
        }

        template <typename _Alloc>
        _Head_base(allocator_arg_t, __uses_alloc1<_Alloc> __a)
            : _Head(allocator_arg, *__a._M_a)
        {
        }

        template <typename _Alloc>
        _Head_base(allocator_arg_t, __uses_alloc2<_Alloc> __a)
            : _Head(*__a._M_a)
        {
        }

        template <typename _UHead>
        _Head_base(__uses_alloc0, _UHead&& __uhead)
            : _Head(std::forward<_UHead>(__uhead))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Head_base(__uses_alloc1<_Alloc> __a, _UHead&& __uhead)
            : _Head(allocator_arg, *__a._M_a, std::forward<_UHead>(__uhead))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Head_base(__uses_alloc2<_Alloc> __a, _UHead&& __uhead)
            : _Head(std::forward<_UHead>(__uhead), *__a._M_a)
        {
        }

        static constexpr _Head& _M_head(_Head_base& __b) noexcept
        {
            return __b;
        }

        static constexpr const _Head& _M_head(const _Head_base& __b) noexcept
        {
            return __b;
        }
    };

    template <std::size_t _Idx, typename _Head>
    struct _Head_base<_Idx, _Head, false>
    {
        constexpr _Head_base()
            : _M_head_impl()
        {
        }

        constexpr _Head_base(const _Head& __h)
            : _M_head_impl(__h)
        {
        }

        constexpr _Head_base(const _Head_base&) = default;
        constexpr _Head_base(_Head_base&&)      = default;

        template <typename _UHead>
        constexpr _Head_base(_UHead&& __h)
            : _M_head_impl(std::forward<_UHead>(__h))
        {
        }

        _Head_base(allocator_arg_t, __uses_alloc0)
            : _M_head_impl()
        {
        }

        template <typename _Alloc>
        _Head_base(allocator_arg_t, __uses_alloc1<_Alloc> __a)
            : _M_head_impl(allocator_arg, *__a._M_a)
        {
        }

        template <typename _Alloc>
        _Head_base(allocator_arg_t, __uses_alloc2<_Alloc> __a)
            : _M_head_impl(*__a._M_a)
        {
        }

        template <typename _UHead>
        _Head_base(__uses_alloc0, _UHead&& __uhead)
            : _M_head_impl(std::forward<_UHead>(__uhead))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Head_base(__uses_alloc1<_Alloc> __a, _UHead&& __uhead)
            : _M_head_impl(allocator_arg, *__a._M_a, std::forward<_UHead>(__uhead))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Head_base(__uses_alloc2<_Alloc> __a, _UHead&& __uhead)
            : _M_head_impl(std::forward<_UHead>(__uhead), *__a._M_a)
        {
        }

        static constexpr _Head& _M_head(_Head_base& __b) noexcept
        {
            return __b._M_head_impl;
        }

        static constexpr const _Head& _M_head(const _Head_base& __b) noexcept
        {
            return __b._M_head_impl;
        }

        _Head _M_head_impl;
    };

    /**
     * Contains the actual implementation of the @c tuple template, stored
     * as a recursive inheritance hierarchy from the first element (most
     * derived class) to the last (least derived class). The @c Idx
     * parameter gives the 0-based index of the element stored at this
     * point in the hierarchy; we use it to implement a constant-time
     * get() operation.
     */
    template <std::size_t _Idx, typename... _Elements>
    struct _Tuple_impl;

    /**
     * Recursive tuple implementation. Here we store the @c Head element
     * and derive from a @c Tuple_impl containing the remaining elements
     * (which contains the @c Tail).
     */
    template <std::size_t _Idx, typename _Head, typename... _Tail>
    struct _Tuple_impl<_Idx, _Head, _Tail...> : public _Tuple_impl<_Idx + 1, _Tail...>,
                                                private _Head_base<_Idx, _Head>
    {
        template <std::size_t, typename...>
        friend class _Tuple_impl;

        typedef _Tuple_impl<_Idx + 1, _Tail...> _Inherited;
        typedef _Head_base<_Idx, _Head>         _Base;

        static constexpr _Head& _M_head(_Tuple_impl& __t) noexcept
        {
            return _Base::_M_head(__t);
        }

        static constexpr const _Head& _M_head(const _Tuple_impl& __t) noexcept
        {
            return _Base::_M_head(__t);
        }

        static constexpr _Inherited& _M_tail(_Tuple_impl& __t) noexcept
        {
            return __t;
        }

        static constexpr const _Inherited& _M_tail(const _Tuple_impl& __t) noexcept
        {
            return __t;
        }

        constexpr _Tuple_impl()
            : _Inherited()
            , _Base()
        {
        }

        explicit constexpr _Tuple_impl(const _Head& __head, const _Tail&... __tail)
            : _Inherited(__tail...)
            , _Base(__head)
        {
        }

        template <typename _UHead,
                  typename... _UTail,
                  typename = typename enable_if<sizeof...(_Tail) == sizeof...(_UTail)>::type>
        explicit constexpr _Tuple_impl(_UHead&& __head, _UTail&&... __tail)
            : _Inherited(std::forward<_UTail>(__tail)...)
            , _Base(std::forward<_UHead>(__head))
        {
        }

        constexpr _Tuple_impl(const _Tuple_impl&) = default;

        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2729. Missing SFINAE on std::pair::operator=
        _Tuple_impl& operator=(const _Tuple_impl&) = delete;

        constexpr _Tuple_impl(_Tuple_impl&& __in) noexcept(
            __and_<is_nothrow_move_constructible<_Head>,
                   is_nothrow_move_constructible<_Inherited>>::value)
            : _Inherited(std::move(_M_tail(__in)))
            , _Base(std::forward<_Head>(_M_head(__in)))
        {
        }

        template <typename... _UElements>
        constexpr _Tuple_impl(const _Tuple_impl<_Idx, _UElements...>& __in)
            : _Inherited(_Tuple_impl<_Idx, _UElements...>::_M_tail(__in))
            , _Base(_Tuple_impl<_Idx, _UElements...>::_M_head(__in))
        {
        }

        template <typename _UHead, typename... _UTails>
        constexpr _Tuple_impl(_Tuple_impl<_Idx, _UHead, _UTails...>&& __in)
            : _Inherited(std::move(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in)))
            , _Base(std::forward<_UHead>(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in)))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a)
            : _Inherited(__tag, __a)
            , _Base(__tag, __use_alloc<_Head>(__a))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag,
                    const _Alloc&   __a,
                    const _Head&    __head,
                    const _Tail&... __tail)
            : _Inherited(__tag, __a, __tail...)
            , _Base(__use_alloc<_Head, _Alloc, _Head>(__a), __head)
        {
        }

        template <typename _Alloc,
                  typename _UHead,
                  typename... _UTail,
                  typename = typename enable_if<sizeof...(_Tail) == sizeof...(_UTail)>::type>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, _UHead&& __head, _UTail&&... __tail)
            : _Inherited(__tag, __a, std::forward<_UTail>(__tail)...)
            , _Base(__use_alloc<_Head, _Alloc, _UHead>(__a), std::forward<_UHead>(__head))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, const _Tuple_impl& __in)
            : _Inherited(__tag, __a, _M_tail(__in))
            , _Base(__use_alloc<_Head, _Alloc, _Head>(__a), _M_head(__in))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, _Tuple_impl&& __in)
            : _Inherited(__tag, __a, std::move(_M_tail(__in)))
            , _Base(__use_alloc<_Head, _Alloc, _Head>(__a), std::forward<_Head>(_M_head(__in)))
        {
        }

        template <typename _Alloc, typename _UHead, typename... _UTails>
        _Tuple_impl(allocator_arg_t                              __tag,
                    const _Alloc&                                __a,
                    const _Tuple_impl<_Idx, _UHead, _UTails...>& __in)
            : _Inherited(__tag, __a, _Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in))
            , _Base(__use_alloc<_Head, _Alloc, const _UHead&>(__a),
                    _Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in))
        {
        }

        template <typename _Alloc, typename _UHead, typename... _UTails>
        _Tuple_impl(allocator_arg_t                         __tag,
                    const _Alloc&                           __a,
                    _Tuple_impl<_Idx, _UHead, _UTails...>&& __in)
            : _Inherited(
                __tag, __a, std::move(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in)))
            , _Base(__use_alloc<_Head, _Alloc, _UHead>(__a),
                    std::forward<_UHead>(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in)))
        {
        }

        template <typename... _UElements>
        void _M_assign(const _Tuple_impl<_Idx, _UElements...>& __in)
        {
            _M_head(*this) = _Tuple_impl<_Idx, _UElements...>::_M_head(__in);
            _M_tail(*this)._M_assign(_Tuple_impl<_Idx, _UElements...>::_M_tail(__in));
        }

        template <typename _UHead, typename... _UTails>
        void _M_assign(_Tuple_impl<_Idx, _UHead, _UTails...>&& __in)
        {
            _M_head(*this)
                = std::forward<_UHead>(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in));
            _M_tail(*this)._M_assign(
                std::move(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in)));
        }

    protected:
        void _M_swap(_Tuple_impl& __in)
        {
            using std::swap;
            swap(_M_head(*this), _M_head(__in));
            _Inherited::_M_swap(_M_tail(__in));
        }
    };

    // Basis case of inheritance recursion.
    template <std::size_t _Idx, typename _Head>
    struct _Tuple_impl<_Idx, _Head> : private _Head_base<_Idx, _Head>
    {
        template <std::size_t, typename...>
        friend class _Tuple_impl;

        typedef _Head_base<_Idx, _Head> _Base;

        static constexpr _Head& _M_head(_Tuple_impl& __t) noexcept
        {
            return _Base::_M_head(__t);
        }

        static constexpr const _Head& _M_head(const _Tuple_impl& __t) noexcept
        {
            return _Base::_M_head(__t);
        }

        constexpr _Tuple_impl()
            : _Base()
        {
        }

        explicit constexpr _Tuple_impl(const _Head& __head)
            : _Base(__head)
        {
        }

        template <typename _UHead>
        explicit constexpr _Tuple_impl(_UHead&& __head)
            : _Base(std::forward<_UHead>(__head))
        {
        }

        constexpr _Tuple_impl(const _Tuple_impl&) = default;

        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2729. Missing SFINAE on std::pair::operator=
        _Tuple_impl& operator=(const _Tuple_impl&) = delete;

        constexpr _Tuple_impl(_Tuple_impl&& __in) noexcept(
            is_nothrow_move_constructible<_Head>::value)
            : _Base(std::forward<_Head>(_M_head(__in)))
        {
        }

        template <typename _UHead>
        constexpr _Tuple_impl(const _Tuple_impl<_Idx, _UHead>& __in)
            : _Base(_Tuple_impl<_Idx, _UHead>::_M_head(__in))
        {
        }

        template <typename _UHead>
        constexpr _Tuple_impl(_Tuple_impl<_Idx, _UHead>&& __in)
            : _Base(std::forward<_UHead>(_Tuple_impl<_Idx, _UHead>::_M_head(__in)))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a)
            : _Base(__tag, __use_alloc<_Head>(__a))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, const _Head& __head)
            : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), __head)
        {
        }

        template <typename _Alloc, typename _UHead>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, _UHead&& __head)
            : _Base(__use_alloc<_Head, _Alloc, _UHead>(__a), std::forward<_UHead>(__head))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, const _Tuple_impl& __in)
            : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), _M_head(__in))
        {
        }

        template <typename _Alloc>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, _Tuple_impl&& __in)
            : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), std::forward<_Head>(_M_head(__in)))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, const _Tuple_impl<_Idx, _UHead>& __in)
            : _Base(__use_alloc<_Head, _Alloc, const _UHead&>(__a),
                    _Tuple_impl<_Idx, _UHead>::_M_head(__in))
        {
        }

        template <typename _Alloc, typename _UHead>
        _Tuple_impl(allocator_arg_t __tag, const _Alloc& __a, _Tuple_impl<_Idx, _UHead>&& __in)
            : _Base(__use_alloc<_Head, _Alloc, _UHead>(__a),
                    std::forward<_UHead>(_Tuple_impl<_Idx, _UHead>::_M_head(__in)))
        {
        }

        template <typename _UHead>
        void _M_assign(const _Tuple_impl<_Idx, _UHead>& __in)
        {
            _M_head(*this) = _Tuple_impl<_Idx, _UHead>::_M_head(__in);
        }

        template <typename _UHead>
        void _M_assign(_Tuple_impl<_Idx, _UHead>&& __in)
        {
            _M_head(*this) = std::forward<_UHead>(_Tuple_impl<_Idx, _UHead>::_M_head(__in));
        }

    protected:
        void _M_swap(_Tuple_impl& __in)
        {
            using std::swap;
            swap(_M_head(*this), _M_head(__in));
        }
    };

    // Concept utility functions, reused in conditionally-explicit
    // constructors.
    template <bool, typename... _Elements>
    struct _TC
    {
        template <typename... _UElements>
        static constexpr bool _ConstructibleTuple()
        {
            return __and_<is_constructible<_Elements, const _UElements&>...>::value;
        }

        template <typename... _UElements>
        static constexpr bool _ImplicitlyConvertibleTuple()
        {
            return __and_<is_convertible<const _UElements&, _Elements>...>::value;
        }

        template <typename... _UElements>
        static constexpr bool _MoveConstructibleTuple()
        {
            return __and_<is_constructible<_Elements, _UElements&&>...>::value;
        }

        template <typename... _UElements>
        static constexpr bool _ImplicitlyMoveConvertibleTuple()
        {
            return __and_<is_convertible<_UElements&&, _Elements>...>::value;
        }

        template <typename _SrcTuple>
        static constexpr bool _NonNestedTuple()
        {
            return __and_<__not_<is_same<tuple<_Elements...>, __remove_cvref_t<_SrcTuple>>>,
                          __not_<is_convertible<_SrcTuple, _Elements...>>,
                          __not_<is_constructible<_Elements..., _SrcTuple>>>::value;
        }

        template <typename... _UElements>
        static constexpr bool _NotSameTuple()
        {
            return __not_<is_same<tuple<_Elements...>, __remove_cvref_t<_UElements>...>>::value;
        }
    };

    template <typename... _Elements>
    struct _TC<false, _Elements...>
    {
        template <typename... _UElements>
        static constexpr bool _ConstructibleTuple()
        {
            return false;
        }

        template <typename... _UElements>
        static constexpr bool _ImplicitlyConvertibleTuple()
        {
            return false;
        }

        template <typename... _UElements>
        static constexpr bool _MoveConstructibleTuple()
        {
            return false;
        }

        template <typename... _UElements>
        static constexpr bool _ImplicitlyMoveConvertibleTuple()
        {
            return false;
        }

        template <typename... _UElements>
        static constexpr bool _NonNestedTuple()
        {
            return true;
        }

        template <typename... _UElements>
        static constexpr bool _NotSameTuple()
        {
            return true;
        }
    };

    /// Primary class template, tuple
    template <typename... _Elements>
    class tuple : public _Tuple_impl<0, _Elements...>
    {
        typedef _Tuple_impl<0, _Elements...> _Inherited;

        // Used for constraining the default constructor so
        // that it becomes dependent on the constraints.
        template <typename _Dummy>
        struct _TC2
        {
            static constexpr bool _DefaultConstructibleTuple()
            {
                return __and_<is_default_constructible<_Elements>...>::value;
            }
            static constexpr bool _ImplicitlyDefaultConstructibleTuple()
            {
                return __and_<__is_implicitly_default_constructible<_Elements>...>::value;
            }
        };

        template <typename... _UElements>
        static constexpr __enable_if_t<sizeof...(_UElements) == sizeof...(_Elements), bool>
            __assignable()
        {
            return __and_<is_assignable<_Elements&, _UElements>...>::value;
        }

        template <typename... _UElements>
        static constexpr bool __nothrow_assignable()
        {
            return __and_<is_nothrow_assignable<_Elements&, _UElements>...>::value;
        }

    public:
        template <
            typename _Dummy = void,
            typename enable_if<_TC2<_Dummy>::_ImplicitlyDefaultConstructibleTuple(), bool>::type
            = true>
        constexpr tuple()
            : _Inherited()
        {
        }

        template <typename _Dummy = void,
                  typename enable_if<_TC2<_Dummy>::_DefaultConstructibleTuple()
                                         && !_TC2<_Dummy>::_ImplicitlyDefaultConstructibleTuple(),
                                     bool>::type
                  = false>
        explicit constexpr tuple()
            : _Inherited()
        {
        }

        // Shortcut for the cases where constructors taking _Elements...
        // need to be constrained.
        template <typename _Dummy>
        using _TCC = _TC<is_same<_Dummy, void>::value, _Elements...>;

        template <typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>()
                          && _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>()
                          && (sizeof...(_Elements) >= 1),
                      bool>::type
                  = true>
        constexpr tuple(const _Elements&... __elements)
            : _Inherited(__elements...)
        {
        }

        template <typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>()
                          && !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>()
                          && (sizeof...(_Elements) >= 1),
                      bool>::type
                  = false>
        explicit constexpr tuple(const _Elements&... __elements)
            : _Inherited(__elements...)
        {
        }

        // Shortcut for the cases where constructors taking _UElements...
        // need to be constrained.
        template <typename... _UElements>
        using _TMC = _TC<(sizeof...(_Elements) == sizeof...(_UElements))
                             && (_TC<(sizeof...(_UElements) == 1),
                                     _Elements...>::template _NotSameTuple<_UElements...>()),
                         _Elements...>;

        // Shortcut for the cases where constructors taking tuple<_UElements...>
        // need to be constrained.
        template <typename... _UElements>
        using _TMCT = _TC<(sizeof...(_Elements) == sizeof...(_UElements))
                              && !is_same<tuple<_Elements...>, tuple<_UElements...>>::value,
                          _Elements...>;

        template <typename... _UElements,
                  typename enable_if<
                      _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && _TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && (sizeof...(_Elements) >= 1),
                      bool>::type
                  = true>
        constexpr tuple(_UElements&&... __elements)
            : _Inherited(std::forward<_UElements>(__elements)...)
        {
        }

        template <typename... _UElements,
                  typename enable_if<
                      _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && !_TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && (sizeof...(_Elements) >= 1),
                      bool>::type
                  = false>
        explicit constexpr tuple(_UElements&&... __elements)
            : _Inherited(std::forward<_UElements>(__elements)...)
        {
        }

        constexpr tuple(const tuple&) = default;

        constexpr tuple(tuple&&) = default;

        // Shortcut for the cases where constructors taking tuples
        // must avoid creating temporaries.
        template <typename _Dummy>
        using _TNTC = _TC<is_same<_Dummy, void>::value && sizeof...(_Elements) == 1, _Elements...>;

        template <
            typename... _UElements,
            typename _Dummy = void,
            typename enable_if<
                _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>()
                    && _TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<_UElements...>()
                    && _TNTC<_Dummy>::template _NonNestedTuple<const tuple<_UElements...>&>(),
                bool>::type
            = true>
        constexpr tuple(const tuple<_UElements...>& __in)
            : _Inherited(static_cast<const _Tuple_impl<0, _UElements...>&>(__in))
        {
        }

        template <
            typename... _UElements,
            typename _Dummy = void,
            typename enable_if<
                _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>()
                    && !_TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<_UElements...>()
                    && _TNTC<_Dummy>::template _NonNestedTuple<const tuple<_UElements...>&>(),
                bool>::type
            = false>
        explicit constexpr tuple(const tuple<_UElements...>& __in)
            : _Inherited(static_cast<const _Tuple_impl<0, _UElements...>&>(__in))
        {
        }

        template <typename... _UElements,
                  typename _Dummy = void,
                  typename enable_if<
                      _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && _TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && _TNTC<_Dummy>::template _NonNestedTuple<tuple<_UElements...>&&>(),
                      bool>::type
                  = true>
        constexpr tuple(tuple<_UElements...>&& __in)
            : _Inherited(static_cast<_Tuple_impl<0, _UElements...>&&>(__in))
        {
        }

        template <typename... _UElements,
                  typename _Dummy = void,
                  typename enable_if<
                      _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && !_TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && _TNTC<_Dummy>::template _NonNestedTuple<tuple<_UElements...>&&>(),
                      bool>::type
                  = false>
        explicit constexpr tuple(tuple<_UElements...>&& __in)
            : _Inherited(static_cast<_Tuple_impl<0, _UElements...>&&>(__in))
        {
        }

        // Allocator-extended constructors.

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a)
            : _Inherited(__tag, __a)
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>()
                          && _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>(),
                      bool>::type
                  = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, const _Elements&... __elements)
            : _Inherited(__tag, __a, __elements...)
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>()
                          && !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>(),
                      bool>::type
                  = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, const _Elements&... __elements)
            : _Inherited(__tag, __a, __elements...)
        {
        }

        template <typename _Alloc,
                  typename... _UElements,
                  typename enable_if<
                      _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && _TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>(),
                      bool>::type
                  = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, _UElements&&... __elements)
            : _Inherited(__tag, __a, std::forward<_UElements>(__elements)...)
        {
        }

        template <typename _Alloc,
                  typename... _UElements,
                  typename enable_if<
                      _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && !_TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>(),
                      bool>::type
                  = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, _UElements&&... __elements)
            : _Inherited(__tag, __a, std::forward<_UElements>(__elements)...)
        {
        }

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple& __in)
            : _Inherited(__tag, __a, static_cast<const _Inherited&>(__in))
        {
        }

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a, tuple&& __in)
            : _Inherited(__tag, __a, static_cast<_Inherited&&>(__in))
        {
        }

        template <
            typename _Alloc,
            typename _Dummy = void,
            typename... _UElements,
            typename enable_if<
                _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>()
                    && _TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<_UElements...>()
                    && _TNTC<_Dummy>::template _NonNestedTuple<const tuple<_UElements...>&>(),
                bool>::type
            = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple<_UElements...>& __in)
            : _Inherited(__tag, __a, static_cast<const _Tuple_impl<0, _UElements...>&>(__in))
        {
        }

        template <
            typename _Alloc,
            typename _Dummy = void,
            typename... _UElements,
            typename enable_if<
                _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>()
                    && !_TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<_UElements...>()
                    && _TNTC<_Dummy>::template _NonNestedTuple<const tuple<_UElements...>&>(),
                bool>::type
            = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple<_UElements...>& __in)
            : _Inherited(__tag, __a, static_cast<const _Tuple_impl<0, _UElements...>&>(__in))
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename... _UElements,
                  typename enable_if<
                      _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && _TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && _TNTC<_Dummy>::template _NonNestedTuple<tuple<_UElements...>&&>(),
                      bool>::type
                  = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, tuple<_UElements...>&& __in)
            : _Inherited(__tag, __a, static_cast<_Tuple_impl<0, _UElements...>&&>(__in))
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename... _UElements,
                  typename enable_if<
                      _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>()
                          && !_TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                              _UElements...>()
                          && _TNTC<_Dummy>::template _NonNestedTuple<tuple<_UElements...>&&>(),
                      bool>::type
                  = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, tuple<_UElements...>&& __in)
            : _Inherited(__tag, __a, static_cast<_Tuple_impl<0, _UElements...>&&>(__in))
        {
        }

        // tuple assignment

        tuple& operator=(typename conditional<__assignable<const _Elements&...>(),
                                              const tuple&,
                                              const __nonesuch_no_braces&>::type
                             __in) noexcept(__nothrow_assignable<const _Elements&...>())
        {
            this->_M_assign(__in);
            return *this;
        }

        tuple& operator=(
            typename conditional<__assignable<_Elements...>(), tuple&&, __nonesuch_no_braces&&>::
                type __in) noexcept(__nothrow_assignable<_Elements...>())
        {
            this->_M_assign(std::move(__in));
            return *this;
        }

        template <typename... _UElements>
        __enable_if_t<__assignable<const _UElements&...>(), tuple&> operator=(
            const tuple<_UElements...>& __in) noexcept(__nothrow_assignable<const _UElements&...>())
        {
            this->_M_assign(__in);
            return *this;
        }

        template <typename... _UElements>
        __enable_if_t<__assignable<_UElements...>(), tuple&>
            operator=(tuple<_UElements...>&& __in) noexcept(__nothrow_assignable<_UElements...>())
        {
            this->_M_assign(std::move(__in));
            return *this;
        }

        // tuple swap
        void swap(tuple& __in) noexcept(__and_<__is_nothrow_swappable<_Elements>...>::value)
        {
            _Inherited::_M_swap(__in);
        }
    };

#if __cpp_deduction_guides >= 201606
    template <typename... _UTypes>
    tuple(_UTypes...) -> tuple<_UTypes...>;
    // template<typename _T1, typename _T2>
    // tuple(pair<_T1, _T2>) -> tuple<_T1, _T2>;
    template <typename _Alloc, typename... _UTypes>
    tuple(allocator_arg_t, _Alloc, _UTypes...) -> tuple<_UTypes...>;
    // template<typename _Alloc, typename _T1, typename _T2>
    // tuple(allocator_arg_t, _Alloc, pair<_T1, _T2>) -> tuple<_T1, _T2>;
    template <typename _Alloc, typename... _UTypes>
    tuple(allocator_arg_t, _Alloc, tuple<_UTypes...>) -> tuple<_UTypes...>;
#endif

    // Explicit specialization, zero-element tuple.
    template <>
    class tuple<>
    {
    public:
        void swap(tuple&) noexcept
        { /* no-op */
        }
        // We need the default since we're going to define no-op
        // allocator constructors.
        tuple() = default;
        // No-op allocator constructors.
        template <typename _Alloc>
        tuple(allocator_arg_t, const _Alloc&)
        {
        }
        template <typename _Alloc>
        tuple(allocator_arg_t, const _Alloc&, const tuple&)
        {
        }
    };

    /// Partial specialization, 2-element tuple.
    /// Includes construction and assignment from a pair.
    template <typename _T1, typename _T2>
    class tuple<_T1, _T2> : public _Tuple_impl<0, _T1, _T2>
    {
        typedef _Tuple_impl<0, _T1, _T2> _Inherited;

        template <typename _U1, typename _U2>
        static constexpr bool __assignable()
        {
            return __and_<is_assignable<_T1&, _U1>, is_assignable<_T2&, _U2>>::value;
        }

        template <typename _U1, typename _U2>
        static constexpr bool __nothrow_assignable()
        {
            return __and_<is_nothrow_assignable<_T1&, _U1>,
                          is_nothrow_assignable<_T2&, _U2>>::value;
        }

    public:
        template <typename _U1 = _T1,
                  typename _U2 = _T2,
                  typename enable_if<__and_<__is_implicitly_default_constructible<_U1>,
                                            __is_implicitly_default_constructible<_U2>>::value,
                                     bool>::type
                  = true>
        constexpr tuple()
            : _Inherited()
        {
        }

        template <typename _U1 = _T1,
                  typename _U2 = _T2,
                  typename enable_if<
                      __and_<is_default_constructible<_U1>,
                             is_default_constructible<_U2>,
                             __not_<__and_<__is_implicitly_default_constructible<_U1>,
                                           __is_implicitly_default_constructible<_U2>>>>::value,
                      bool>::type
                  = false>
        explicit constexpr tuple()
            : _Inherited()
        {
        }

        // Shortcut for the cases where constructors taking _T1, _T2
        // need to be constrained.
        template <typename _Dummy>
        using _TCC = _TC<is_same<_Dummy, void>::value, _T1, _T2>;

        template <typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>()
                          && _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                      bool>::type
                  = true>
        constexpr tuple(const _T1& __a1, const _T2& __a2)
            : _Inherited(__a1, __a2)
        {
        }

        template <typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>()
                          && !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                      bool>::type
                  = false>
        explicit constexpr tuple(const _T1& __a1, const _T2& __a2)
            : _Inherited(__a1, __a2)
        {
        }

        // Shortcut for the cases where constructors taking _U1, _U2
        // need to be constrained.
        using _TMC = _TC<true, _T1, _T2>;

        template <
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>()
                                   && !is_same<__remove_cvref_t<_U1>, allocator_arg_t>::value,
                               bool>::type
            = true>
        constexpr tuple(_U1&& __a1, _U2&& __a2)
            : _Inherited(std::forward<_U1>(__a1), std::forward<_U2>(__a2))
        {
        }

        template <
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>()
                                   && !is_same<__remove_cvref_t<_U1>, allocator_arg_t>::value,
                               bool>::type
            = false>
        explicit constexpr tuple(_U1&& __a1, _U2&& __a2)
            : _Inherited(std::forward<_U1>(__a1), std::forward<_U2>(__a2))
        {
        }

        constexpr tuple(const tuple&) = default;

        constexpr tuple(tuple&&) = default;

        template <typename _U1,
                  typename _U2,
                  typename enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
                                         && _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                                     bool>::type
                  = true>
        constexpr tuple(const tuple<_U1, _U2>& __in)
            : _Inherited(static_cast<const _Tuple_impl<0, _U1, _U2>&>(__in))
        {
        }

        template <typename _U1,
                  typename _U2,
                  typename enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
                                         && !_TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                                     bool>::type
                  = false>
        explicit constexpr tuple(const tuple<_U1, _U2>& __in)
            : _Inherited(static_cast<const _Tuple_impl<0, _U1, _U2>&>(__in))
        {
        }

        template <
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = true>
        constexpr tuple(tuple<_U1, _U2>&& __in)
            : _Inherited(static_cast<_Tuple_impl<0, _U1, _U2>&&>(__in))
        {
        }

        template <
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = false>
        explicit constexpr tuple(tuple<_U1, _U2>&& __in)
            : _Inherited(static_cast<_Tuple_impl<0, _U1, _U2>&&>(__in))
        {
        }

        // Allocator-extended constructors.

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a)
            : _Inherited(__tag, __a)
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>()
                          && _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                      bool>::type
                  = true>

        tuple(allocator_arg_t __tag, const _Alloc& __a, const _T1& __a1, const _T2& __a2)
            : _Inherited(__tag, __a, __a1, __a2)
        {
        }

        template <typename _Alloc,
                  typename _Dummy = void,
                  typename enable_if<
                      _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>()
                          && !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                      bool>::type
                  = false>

        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, const _T1& __a1, const _T2& __a2)
            : _Inherited(__tag, __a, __a1, __a2)
        {
        }

        template <
            typename _Alloc,
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, _U1&& __a1, _U2&& __a2)
            : _Inherited(__tag, __a, std::forward<_U1>(__a1), std::forward<_U2>(__a2))
        {
        }

        template <
            typename _Alloc,
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, _U1&& __a1, _U2&& __a2)
            : _Inherited(__tag, __a, std::forward<_U1>(__a1), std::forward<_U2>(__a2))
        {
        }

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple& __in)
            : _Inherited(__tag, __a, static_cast<const _Inherited&>(__in))
        {
        }

        template <typename _Alloc>
        tuple(allocator_arg_t __tag, const _Alloc& __a, tuple&& __in)
            : _Inherited(__tag, __a, static_cast<_Inherited&&>(__in))
        {
        }

        template <typename _Alloc,
                  typename _U1,
                  typename _U2,
                  typename enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
                                         && _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                                     bool>::type
                  = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple<_U1, _U2>& __in)
            : _Inherited(__tag, __a, static_cast<const _Tuple_impl<0, _U1, _U2>&>(__in))
        {
        }

        template <typename _Alloc,
                  typename _U1,
                  typename _U2,
                  typename enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
                                         && !_TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                                     bool>::type
                  = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, const tuple<_U1, _U2>& __in)
            : _Inherited(__tag, __a, static_cast<const _Tuple_impl<0, _U1, _U2>&>(__in))
        {
        }

        template <
            typename _Alloc,
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = true>
        tuple(allocator_arg_t __tag, const _Alloc& __a, tuple<_U1, _U2>&& __in)
            : _Inherited(__tag, __a, static_cast<_Tuple_impl<0, _U1, _U2>&&>(__in))
        {
        }

        template <
            typename _Alloc,
            typename _U1,
            typename _U2,
            typename enable_if<_TMC::template _MoveConstructibleTuple<_U1, _U2>()
                                   && !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                               bool>::type
            = false>
        explicit tuple(allocator_arg_t __tag, const _Alloc& __a, tuple<_U1, _U2>&& __in)
            : _Inherited(__tag, __a, static_cast<_Tuple_impl<0, _U1, _U2>&&>(__in))
        {
        }

        tuple& operator=(typename conditional<__assignable<const _T1&, const _T2&>(),
                                              const tuple&,
                                              const __nonesuch_no_braces&>::type
                             __in) noexcept(__nothrow_assignable<const _T1&, const _T2&>())
        {
            this->_M_assign(__in);
            return *this;
        }

        tuple& operator=(
            typename conditional<__assignable<_T1, _T2>(), tuple&&, __nonesuch_no_braces&&>::type
                __in) noexcept(__nothrow_assignable<_T1, _T2>())
        {
            this->_M_assign(std::move(__in));
            return *this;
        }

        template <typename _U1, typename _U2>
        __enable_if_t<__assignable<const _U1&, const _U2&>(), tuple&> operator=(
            const tuple<_U1, _U2>& __in) noexcept(__nothrow_assignable<const _U1&, const _U2&>())
        {
            this->_M_assign(__in);
            return *this;
        }

        template <typename _U1, typename _U2>
        __enable_if_t<__assignable<_U1, _U2>(), tuple&>
            operator=(tuple<_U1, _U2>&& __in) noexcept(__nothrow_assignable<_U1, _U2>())
        {
            this->_M_assign(std::move(__in));
            return *this;
        }

        void swap(tuple& __in) noexcept(
            __and_<__is_nothrow_swappable<_T1>, __is_nothrow_swappable<_T2>>::value)
        {
            _Inherited::_M_swap(__in);
        }
    };

    /// class tuple_size
    template <typename... _Elements>
    struct tuple_size<tuple<_Elements...>>
        : public integral_constant<std::size_t, sizeof...(_Elements)>
    {
    };

    template <typename _Tp>
    inline constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

    /**
     * Recursive case for tuple_element: strip off the first element in
     * the tuple and retrieve the (i-1)th element of the remaining tuple.
     */
    template <std::size_t __i, typename _Head, typename... _Tail>
    struct tuple_element<__i, tuple<_Head, _Tail...>> : tuple_element<__i - 1, tuple<_Tail...>>
    {
    };

    /**
     * Basis case for tuple_element: The first element is the one we're seeking.
     */
    template <typename _Head, typename... _Tail>
    struct tuple_element<0, tuple<_Head, _Tail...>>
    {
        typedef _Head type;
    };

    /**
     * Error case for tuple_element: invalid index.
     */
    template <size_t __i>
    struct tuple_element<__i, tuple<>>
    {
        static_assert(__i < tuple_size<tuple<>>::value, "tuple index is in range");
    };

    // Duplicate of C++14's tuple_element_t for internal use in C++11 mode
    template <std::size_t __i, typename _Tp>
    using __tuple_element_t = typename tuple_element<__i, _Tp>::type;

    template <std::size_t __i, typename _Tp>
    using tuple_element_t = typename tuple_element<__i, _Tp>::type;

    template <std::size_t __i, typename _Head, typename... _Tail>
    constexpr _Head& __get_helper(_Tuple_impl<__i, _Head, _Tail...>& __t) noexcept
    {
        return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
    }

    template <std::size_t __i, typename _Head, typename... _Tail>
    constexpr const _Head& __get_helper(const _Tuple_impl<__i, _Head, _Tail...>& __t) noexcept
    {
        return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
    }

    /// Return a reference to the ith element of a tuple.
    template <std::size_t __i, typename... _Elements>
    constexpr __tuple_element_t<__i, tuple<_Elements...>>& get(tuple<_Elements...>& __t) noexcept
    {
        return std::__get_helper<__i>(__t);
    }

    /// Return a const reference to the ith element of a const tuple.
    template <std::size_t __i, typename... _Elements>
    constexpr const __tuple_element_t<__i, tuple<_Elements...>>&
        get(const tuple<_Elements...>& __t) noexcept
    {
        return std::__get_helper<__i>(__t);
    }

    /// Return an rvalue reference to the ith element of a tuple rvalue.
    template <std::size_t __i, typename... _Elements>
    constexpr __tuple_element_t<__i, tuple<_Elements...>>&& get(tuple<_Elements...>&& __t) noexcept
    {
        typedef __tuple_element_t<__i, tuple<_Elements...>> __element_type;
        return std::forward<__element_type&&>(std::get<__i>(__t));
    }

    /// Return a const rvalue reference to the ith element of a const tuple rvalue.
    template <std::size_t __i, typename... _Elements>
    constexpr const __tuple_element_t<__i, tuple<_Elements...>>&&
        get(const tuple<_Elements...>&& __t) noexcept
    {
        typedef __tuple_element_t<__i, tuple<_Elements...>> __element_type;
        return std::forward<const __element_type&&>(std::get<__i>(__t));
    }

#if __cplusplus >= 201402L

    template <typename _Head, size_t __i, typename... _Tail>
    constexpr _Head& __get_helper2(_Tuple_impl<__i, _Head, _Tail...>& __t) noexcept
    {
        return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
    }

    template <typename _Head, size_t __i, typename... _Tail>
    constexpr const _Head& __get_helper2(const _Tuple_impl<__i, _Head, _Tail...>& __t) noexcept
    {
        return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
    }

    /// Return a reference to the unique element of type _Tp of a tuple.
    template <typename _Tp, typename... _Types>
    constexpr _Tp& get(tuple<_Types...>& __t) noexcept
    {
        return std::__get_helper2<_Tp>(__t);
    }

    /// Return a reference to the unique element of type _Tp of a tuple rvalue.
    template <typename _Tp, typename... _Types>
    constexpr _Tp&& get(tuple<_Types...>&& __t) noexcept
    {
        return std::forward<_Tp&&>(std::__get_helper2<_Tp>(__t));
    }

    /// Return a const reference to the unique element of type _Tp of a tuple.
    template <typename _Tp, typename... _Types>
    constexpr const _Tp& get(const tuple<_Types...>& __t) noexcept
    {
        return std::__get_helper2<_Tp>(__t);
    }

    /// Return a const reference to the unique element of type _Tp of
    /// a const tuple rvalue.
    template <typename _Tp, typename... _Types>
    constexpr const _Tp&& get(const tuple<_Types...>&& __t) noexcept
    {
        return std::forward<const _Tp&&>(std::__get_helper2<_Tp>(__t));
    }
#endif

    // This class performs the comparison operations on tuples
    template <typename _Tp, typename _Up, size_t __i, size_t __size>
    struct __tuple_compare
    {
        static constexpr bool __eq(const _Tp& __t, const _Up& __u)
        {
            return bool(std::get<__i>(__t) == std::get<__i>(__u))
                   && __tuple_compare<_Tp, _Up, __i + 1, __size>::__eq(__t, __u);
        }

        static constexpr bool __less(const _Tp& __t, const _Up& __u)
        {
            return bool(std::get<__i>(__t) < std::get<__i>(__u))
                   || (!bool(std::get<__i>(__u) < std::get<__i>(__t))
                       && __tuple_compare<_Tp, _Up, __i + 1, __size>::__less(__t, __u));
        }
    };

    template <typename _Tp, typename _Up, size_t __size>
    struct __tuple_compare<_Tp, _Up, __size, __size>
    {
        static constexpr bool __eq(const _Tp&, const _Up&)
        {
            return true;
        }

        static constexpr bool __less(const _Tp&, const _Up&)
        {
            return false;
        }
    };

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator==(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                      "tuple objects can only be compared if they have equal sizes.");
        using __compare
            = __tuple_compare<tuple<_TElements...>, tuple<_UElements...>, 0, sizeof...(_TElements)>;
        return __compare::__eq(__t, __u);
    }

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator<(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                      "tuple objects can only be compared if they have equal sizes.");
        using __compare
            = __tuple_compare<tuple<_TElements...>, tuple<_UElements...>, 0, sizeof...(_TElements)>;
        return __compare::__less(__t, __u);
    }

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator!=(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        return !(__t == __u);
    }

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator>(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        return __u < __t;
    }

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator<=(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        return !(__u < __t);
    }

    template <typename... _TElements, typename... _UElements>
    constexpr bool operator>=(const tuple<_TElements...>& __t, const tuple<_UElements...>& __u)
    {
        return !(__t < __u);
    }

    // NB: DR 705.
    template <typename... _Elements>
    constexpr tuple<typename __decay_and_strip<_Elements>::__type...>
        make_tuple(_Elements&&... __args)
    {
        typedef tuple<typename __decay_and_strip<_Elements>::__type...> __result_type;
        return __result_type(std::forward<_Elements>(__args)...);
    }

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 2275. Why is forward_as_tuple not constexpr?
    /// std::forward_as_tuple
    template <typename... _Elements>
    constexpr tuple<_Elements&&...> forward_as_tuple(_Elements&&... __args) noexcept
    {
        return tuple<_Elements&&...>(std::forward<_Elements>(__args)...);
    }

    template <size_t, typename, typename, size_t>
    struct __make_tuple_impl;

    template <size_t _Idx, typename _Tuple, typename... _Tp, size_t _Nm>
    struct __make_tuple_impl<_Idx, tuple<_Tp...>, _Tuple, _Nm>
        : __make_tuple_impl<_Idx + 1, tuple<_Tp..., __tuple_element_t<_Idx, _Tuple>>, _Tuple, _Nm>
    {
    };

    template <std::size_t _Nm, typename _Tuple, typename... _Tp>
    struct __make_tuple_impl<_Nm, tuple<_Tp...>, _Tuple, _Nm>
    {
        typedef tuple<_Tp...> __type;
    };

    template <typename _Tuple>
    struct __do_make_tuple : __make_tuple_impl<0, tuple<>, _Tuple, std::tuple_size<_Tuple>::value>
    {
    };

    // Returns the std::tuple equivalent of a tuple-like type.
    template <typename _Tuple>
    struct __make_tuple : public __do_make_tuple<__remove_cvref_t<_Tuple>>
    {
    };

    // Combines several std::tuple's into a single one.
    template <typename...>
    struct __combine_tuples;

    template <>
    struct __combine_tuples<>
    {
        typedef tuple<> __type;
    };

    template <typename... _Ts>
    struct __combine_tuples<tuple<_Ts...>>
    {
        typedef tuple<_Ts...> __type;
    };

    template <typename... _T1s, typename... _T2s, typename... _Rem>
    struct __combine_tuples<tuple<_T1s...>, tuple<_T2s...>, _Rem...>
    {
        typedef typename __combine_tuples<tuple<_T1s..., _T2s...>, _Rem...>::__type __type;
    };

    // Computes the result type of tuple_cat given a set of tuple-like types.
    template <typename... _Tpls>
    struct __tuple_cat_result
    {
        typedef typename __combine_tuples<typename __make_tuple<_Tpls>::__type...>::__type __type;
    };

    // Helper to determine the index set for the first tuple-like
    // type of a given set.
    template <typename...>
    struct __make_1st_indices;

    template <>
    struct __make_1st_indices<>
    {
        typedef std::_Index_tuple<> __type;
    };

    template <typename _Tp, typename... _Tpls>
    struct __make_1st_indices<_Tp, _Tpls...>
    {
        typedef typename std::_Build_index_tuple<
            std::tuple_size<typename std::remove_reference<_Tp>::type>::value>::__type __type;
    };

    // Performs the actual concatenation by step-wise expanding tuple-like
    // objects into the elements, which are finally forwarded into the
    // result tuple.
    template <typename _Ret, typename _Indices, typename... _Tpls>
    struct __tuple_concater;

    template <typename _Ret, std::size_t... _Is, typename _Tp, typename... _Tpls>
    struct __tuple_concater<_Ret, std::_Index_tuple<_Is...>, _Tp, _Tpls...>
    {
        template <typename... _Us>
        static constexpr _Ret _S_do(_Tp&& __tp, _Tpls&&... __tps, _Us&&... __us)
        {
            typedef typename __make_1st_indices<_Tpls...>::__type __idx;
            typedef __tuple_concater<_Ret, __idx, _Tpls...>       __next;
            return __next::_S_do(std::forward<_Tpls>(__tps)...,
                                 std::forward<_Us>(__us)...,
                                 std::get<_Is>(std::forward<_Tp>(__tp))...);
        }
    };

    template <typename _Ret>
    struct __tuple_concater<_Ret, std::_Index_tuple<>>
    {
        template <typename... _Us>
        static constexpr _Ret _S_do(_Us&&... __us)
        {
            return _Ret(std::forward<_Us>(__us)...);
        }
    };

    /// tuple_cat
    template <typename... _Tpls,
              typename = typename enable_if<__and_<__is_tuple_like<_Tpls>...>::value>::type>
    constexpr auto tuple_cat(_Tpls&&... __tpls) -> typename __tuple_cat_result<_Tpls...>::__type
    {
        typedef typename __tuple_cat_result<_Tpls...>::__type __ret;
        typedef typename __make_1st_indices<_Tpls...>::__type __idx;
        typedef __tuple_concater<__ret, __idx, _Tpls...>      __concater;
        return __concater::_S_do(std::forward<_Tpls>(__tpls)...);
    }

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 2301. Why is tie not constexpr?
    /// tie
    template <typename... _Elements>
    constexpr tuple<_Elements&...> tie(_Elements&... __args) noexcept
    {
        return tuple<_Elements&...>(__args...);
    }

    /// swap
    template <typename... _Elements>
    inline
#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
        // Constrained free swap overload, see p0185r1
        typename enable_if<__and_<__is_swappable<_Elements>...>::value>::type
#else
        void
#endif
        swap(tuple<_Elements...>& __x, tuple<_Elements...>& __y) noexcept(noexcept(__x.swap(__y)))
    {
        __x.swap(__y);
    }

#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
    template <typename... _Elements>
    typename enable_if<!__and_<__is_swappable<_Elements>...>::value>::type
        swap(tuple<_Elements...>&, tuple<_Elements...>&)
        = delete;
#endif

    /// Partial specialization for tuples
    template <typename... _Types, typename _Alloc>
    struct uses_allocator<tuple<_Types...>, _Alloc> : true_type
    {
    };

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    constexpr decltype(auto) __apply_impl(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        return std::__invoke(std::forward<_Fn>(__f), std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    template <typename _Fn, typename _Tuple>
    constexpr decltype(auto) apply(_Fn&& __f, _Tuple&& __t)
    {
        using _Indices = std::make_index_sequence<tuple_size_v<remove_reference_t<_Tuple>>>;
        return std::__apply_impl(std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
    }

    template <typename _Tp, typename _Tuple, size_t... _Idx>
    constexpr _Tp __make_from_tuple_impl(_Tuple&& __t, index_sequence<_Idx...>)
    {
        return _Tp(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    template <typename _Tp, typename _Tuple>
    constexpr _Tp make_from_tuple(_Tuple&& __t)
    {
        return __make_from_tuple_impl<_Tp>(
            std::forward<_Tuple>(__t),
            std::make_index_sequence<tuple_size_v<remove_reference_t<_Tuple>>>{});
    }
}

#endif // #if defined(__HIPCC_RTC__)

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
                                  std::make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename... TypesL, typename... TypesR>
    constexpr static inline auto operator*(tuple<TypesL...> const& lhs, tuple<TypesR...> const& rhs)
    {
        return operator_mult_impl(std::forward<decltype(lhs)>(lhs),
                                  std::forward<decltype(rhs)>(rhs),
                                  std::make_index_sequence<tuple_size<tuple<TypesL...>>::value>());
    }

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
                                 std::make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator+(tuple<Types...> const& lhs, T&& rhs)
    {
        return operator_add_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 std::make_index_sequence<tuple_size<tuple<Types...>>::value>());
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
        return make_tuple(get<Indices>(lhs) - std::forward<T>(rhs)...);
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
                                 std::make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

    template <typename T, typename... Types>
    constexpr static inline auto operator-(tuple<Types...> const& lhs, T&& rhs)
    {
        return operator_sub_impl(std::forward<decltype(lhs)>(lhs),
                                 std::forward<decltype(rhs)>(rhs),
                                 std::make_index_sequence<tuple_size<tuple<Types...>>::value>());
    }

} // namespace std

namespace rocwmma
{

    struct MakeTuple
    {
        template <typename... ArgsT>
        auto operator()(ArgsT&&... args)
        {
            return std::make_tuple(std::forward<ArgsT>(args)...);
        }
    };

    template <typename T, std::size_t... Indices>
    constexpr static auto copy_impl(T&& t, std::index_sequence<Indices...>&&)
    {
        return std::make_tuple(get<Indices>(std::forward<T>(t))...);
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

    template <typename T>
    constexpr static auto get_first(T&& t)
    {
        return get<0>(std::forward<T>(t));
    }

    template <typename T>
    constexpr static auto get_last(T&& t)
    {
        return get<std::tuple_size<std::decay_t<decltype(t)>>::value - 1>(std::forward<T>(t));
    }

    template <typename T, std::size_t... Indices>
    constexpr static auto reverse_impl(T&& t, std::index_sequence<Indices...>)
    {
        return std::make_tuple(get<sizeof...(Indices) - 1 - Indices>(std::forward<T>(t))...);
    }

    template <typename T>
    constexpr static auto reverse(T&& t)
    {
        return reverse_impl(std::forward<T>(t),
                            std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
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
        return (flatten(get<Indices>(coord), get<Indices>(dims), mult) + ...);
    }

    template <typename Lhs, typename Rhs>
    constexpr static auto flatten_coord_right(Lhs&& coord, Rhs&& dims)
    {
        return flatten_coord_right_impl(
            std::forward<Lhs>(coord),
            std::forward<Rhs>(dims),
            std::make_index_sequence<std::tuple_size<std::decay_t<Lhs>>::value>());
    }

    template <typename Lhs, typename Rhs, std::size_t... Indices>
    constexpr static auto
        flatten_coord_left_impl(Lhs&& coord, Rhs&& dims, std::index_sequence<Indices...>)
    {
        auto flatten = [](auto&& c, auto&& d, auto& mul) {
            auto result = c * mul;
            mul *= d;
            return result;
        };
        auto mult = 1;
        return (flatten(get<sizeof...(Indices) - 1 - Indices>(std::forward<Lhs>(coord)),
                        get<sizeof...(Indices) - 1 - Indices>(std::forward<Rhs>(dims)),
                        std::forward<decltype(mult)&>(mult))
                + ...);
    }

    template <typename Lhs, typename Rhs>
    constexpr static auto flatten_coord_left(Lhs&& coord, Rhs&& dims)
    {
        return flatten_coord_left_impl(
            std::forward<Lhs>(coord),
            std::forward<Rhs>(dims),
            std::make_index_sequence<std::tuple_size<std::decay_t<Lhs>>::value>());
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
        return std::decay_t<T>{inflate(std::forward<Coord1d const&>(flatCoord),
                                       get<Indices>(std::forward<T>(dims)),
                                       std::forward<decltype(div)&>(div),
                                       Indices == sizeof...(Indices) - 1)...};
    }

    template <typename Coord1d, typename T>
    constexpr static inline auto inflate_coord_right(Coord1d const& flatCoord, T&& dims)
    {
        auto result = inflate_coord_right_impl(
            std::forward<decltype(flatCoord)>(flatCoord),
            std::forward<T>(dims),
            std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>());
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
                    get<std::tuple_size<std::decay_t<T>>::value - 1 - Indices>(dims),
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
            (inflate(get<Indices>(strides), get<Indices>(strideCounts)) + ...)};
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

#if !defined(__HIPCC_RTC__)
    template <class T, size_t... I>
    auto& print(std::ostream& os, T&& t, std::index_sequence<I...>&&)
    {
        os << "(";
        (..., (os << (I == 0 ? "" : ", ") << get<I>(std::forward<T>(t))));
        return os << ")\n";
    }

    template <class... ArgsT>
    auto& print(std::ostream& os, std::tuple<ArgsT...> const& t)
    {
        return print(os, t, std::make_index_sequence<sizeof...(ArgsT)>());
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
