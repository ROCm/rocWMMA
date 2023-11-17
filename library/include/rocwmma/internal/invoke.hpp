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
#ifndef ROCWMMA_INVOKE_HPP
#define ROCWMMA_INVOKE_HPP

#include "type_traits.hpp"

namespace std
{
    // Used by __invoke_impl instead of std::forward<_Tp> so that a
    // reference_wrapper is converted to an lvalue-reference.
    template <typename _Tp, typename _Up = typename __inv_unwrap<_Tp>::type>
    constexpr _Up&& __invfwd(typename remove_reference<_Tp>::type& __t) noexcept
    {
        return static_cast<_Up&&>(__t);
    }

    template <typename _Res, typename _Fn, typename... _Args>
    constexpr _Res __invoke_impl(__invoke_other, _Fn&& __f, _Args&&... __args)
    {
        return std::forward<_Fn>(__f)(std::forward<_Args>(__args)...);
    }

    template <typename _Res, typename _MemFun, typename _Tp, typename... _Args>
    constexpr _Res __invoke_impl(__invoke_memfun_ref, _MemFun&& __f, _Tp&& __t, _Args&&... __args)
    {
        return (__invfwd<_Tp>(__t).*__f)(std::forward<_Args>(__args)...);
    }

    template <typename _Res, typename _MemFun, typename _Tp, typename... _Args>
    constexpr _Res __invoke_impl(__invoke_memfun_deref, _MemFun&& __f, _Tp&& __t, _Args&&... __args)
    {
        return ((*std::forward<_Tp>(__t)).*__f)(std::forward<_Args>(__args)...);
    }

    template <typename _Res, typename _MemPtr, typename _Tp>
    constexpr _Res __invoke_impl(__invoke_memobj_ref, _MemPtr&& __f, _Tp&& __t)
    {
        return __invfwd<_Tp>(__t).*__f;
    }

    template <typename _Res, typename _MemPtr, typename _Tp>
    constexpr _Res __invoke_impl(__invoke_memobj_deref, _MemPtr&& __f, _Tp&& __t)
    {
        return (*std::forward<_Tp>(__t)).*__f;
    }

    /// Invoke a callable object.
    template <typename _Callable, typename... _Args>
    constexpr typename __invoke_result<_Callable, _Args...>::type
        __invoke(_Callable&& __fn,
                 _Args&&... __args) noexcept(__is_nothrow_invocable<_Callable, _Args...>::value)
    {
        using __result = __invoke_result<_Callable, _Args...>;
        using __type   = typename __result::type;
        using __tag    = typename __result::__invoke_type;
        return std::__invoke_impl<__type>(
            __tag{}, std::forward<_Callable>(__fn), std::forward<_Args>(__args)...);
    }
}

#endif // ROCWMMA_INVOKE_HPP
