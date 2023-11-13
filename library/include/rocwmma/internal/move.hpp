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
#ifndef ROCWMMA_MOVE_HPP
#define ROCWMMA_MOVE_HPP

#include "type_traits.hpp"

namespace std
{
    template<typename _Tp>
    inline constexpr _Tp* __addressof(_Tp& __r) noexcept
    { 
        return __builtin_addressof(__r); 
    }

    template<typename _Tp>
    constexpr typename std::remove_reference<_Tp>::type&&
    move(_Tp&& __t) noexcept
    { 
        return static_cast<typename std::remove_reference<_Tp>::type&&>(__t); 
    }


    template<typename _Tp>
    struct __move_if_noexcept_cond
    : public __and_<__not_<is_nothrow_move_constructible<_Tp>>,
                    is_copy_constructible<_Tp>>::type { };

    template<typename _Tp>
    constexpr typename
    conditional<__move_if_noexcept_cond<_Tp>::value, const _Tp&, _Tp&&>::type
    move_if_noexcept(_Tp& __x) noexcept
    { 
        return std::move(__x); 
    }

    template<typename _Tp>
    inline constexpr _Tp*
    addressof(_Tp& __r) noexcept
    { 
        return std::__addressof(__r); 
    }

    template<typename _Tp>
    const _Tp* addressof(const _Tp&&) = delete;

    // C++11 version of std::exchange for internal use.
    template <typename _Tp, typename _Up = _Tp>
    inline _Tp
    __exchange(_Tp& __obj, _Up&& __new_val)
    {
        _Tp __old_val = std::move(__obj);
        __obj = std::forward<_Up>(__new_val);
        return __old_val;
    }

    template<typename _Tp>
    inline typename enable_if<__and_<__not_<__is_tuple_like<_Tp>>,
			      is_move_constructible<_Tp>,
			      is_move_assignable<_Tp>>::value>::type
    swap(_Tp& __a, _Tp& __b)
        noexcept(__and_<is_nothrow_move_constructible<_Tp>,
	            is_nothrow_move_assignable<_Tp>>::value)
    {
        _Tp __tmp = move(__a);
        __a = move(__b);
        __b = move(__tmp);
    }

    // /// Swap the contents of two arrays.
    // template<typename _Tp, size_t _Nm>
    // inline typename enable_if<__is_swappable<_Tp>::value>::type
    // swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
    //     noexcept(__is_nothrow_swappable<_Tp>::value)
    // {
    //     for (size_t __n = 0; __n < _Nm; ++__n)
	//         swap(__a[__n], __b[__n]);
    // }
}

#endif // ROCWMMA_MOVE_HPP