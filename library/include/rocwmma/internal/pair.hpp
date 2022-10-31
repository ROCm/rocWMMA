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
#ifndef ROCWMMA_PAIR_HPP
#define ROCWMMA_PAIR_HPP

namespace rocwmma
{
    template <typename T0,
              typename T1,
              class = typename std::enable_if<std::is_same<T0, T1>::value>::type>
    using rocwmma_pair = HIP_vector_type<T0, 2>;

    namespace detail
    {
        template <class T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T> make_pair(T&& x, T&& y)
        {
            return rocwmma_pair<T, T>(x, y);
        }

        template <uint32_t idx, typename T>
        __host__ __device__ constexpr static inline T& get(rocwmma_pair<T, T>& pair)
        {
            if(idx == 0)
                return pair.x;
            else if(idx == 1)
                return pair.y;
        }

        template <uint32_t idx, typename T>
        __host__ __device__ constexpr static inline T get(rocwmma_pair<T, T> const& pair)
        {
            if(idx == 0)
                return pair.x;
            else if(idx == 1)
                return pair.y;
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T> swap(T const& x, T const& y)
        {
            return rocwmma_pair<T, T>(y, x);
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T> swap(T& x, T& y)
        {
            return rocwmma_pair<T, T>(y, x);
        }

        // Single operand for swap
        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>
                 swap(rocwmma_pair<T, T> const& p)
        {
            return make_pair(get<1>(p), get<0>(p));
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>& swap(rocwmma_pair<T, T>& p)
        {
            swap(get<0>(p), get<1>(p));
            return p;
        }

        // Add, sub operators
        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>
                 operator+(rocwmma_pair<T, T> const& lhs, rocwmma_pair<T, T> const& rhs)
        {
            return make_pair(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>&
                 operator+=(rocwmma_pair<T, T>& lhs, rocwmma_pair<T, T> const& rhs)
        {
            get<0>(lhs) += get<0>(rhs);
            get<1>(lhs) += get<1>(rhs);
            return lhs;
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>
                 operator*(rocwmma_pair<T, T> const& lhs, rocwmma_pair<T, T> const& rhs)
        {
            return make_pair(get<0>(lhs) * get<0>(rhs), get<1>(lhs) * get<1>(rhs));
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>&
                 operator*=(rocwmma_pair<T, T>& lhs, rocwmma_pair<T, T> const& rhs)
        {
            get<0>(lhs) *= get<0>(rhs);
            get<1>(lhs) *= get<1>(rhs);
            return lhs;
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>
                 operator-(rocwmma_pair<T, T> const& lhs, rocwmma_pair<T, T> const& rhs)
        {
            return make_pair(get<0>(lhs) - get<0>(rhs), get<1>(lhs) - get<1>(rhs));
        }

        template <typename T>
        __host__ __device__ constexpr static inline rocwmma_pair<T, T>&
                 operator-=(rocwmma_pair<T, T>& lhs, rocwmma_pair<T, T> const& rhs)
        {
            get<0>(lhs) -= get<0>(rhs);
            get<1>(lhs) -= get<1>(rhs);
            return lhs;
        }
    } // namespace detail
} // namespace rocwmma

#endif // ROCWMMA_PAIR_HPP
