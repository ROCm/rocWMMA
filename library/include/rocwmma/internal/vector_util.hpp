/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_VECTOR_UTIL_HPP
#define ROCWMMA_VECTOR_UTIL_HPP

#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{
    //! Extracts the first (lo) half of elements from a given vector
    /*!
      \param v Vector to extract the lo elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractLo(VecT<DataT, VecSize> const& v);

    //! Extracts the second (hi) half of elements from a given vector
    /*!
      \param v Vector to extract the hi elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractHi(VecT<DataT, VecSize> const& v);

    //! Extracts the the even elements elements from a given vector
    /*!
      \param v Vector to extract the even elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v);

    //! Extracts the the odd elements elements from a given vector
    /*!
      \param v Vector to extract the odd elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v);

    //! Re-orders vector elements such that even elements are concatenated with odd elements.
    /*!
      \param v Vector to reorder elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderEvenOdd(VecT<DataT, VecSize> const& v);

    //! Re-orders vector elements such that odd elements are concatenated with even elements.
    /*!
      \param v Vector to reorder elements from.
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderOddEven(VecT<DataT, VecSize> const& v);

    //! Concatenates the contents of two vectors together in order.
    /*!
      \param v0 First vector to concatenate
      \param v1 Second vector to concatenate
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                       VecT<DataT, VecSize> const& v1);

    //! Alternates selecting even elements from the first vector and odd elements from the second vector.
    //! Analogous to a zipper.
    //! E.g.
    //! v0     = [0, 1]
    //! v1     = [2, 3]
    //! result = [0, 3]
    /*!
      \param v0 Vector from which even elements are alternately selected
      \param v1 Vector from which odd elements are alternately selected
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                    VecT<DataT, VecSize> const& v1);

    //! Alternates selecting the first (lo) half of elements from each vector
    //! E.g.
    //! v0     = [0, 1]
    //! v1     = [2, 3]
    //! result = [0, 2]
    /*!
      \param v0 Vector from which lo elements are alternately selected
      \param v1 Vector from which lo elements are alternately selected
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackLo(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1);

    //! Alternates selecting the second (hi) half of elements from each vector
    //! E.g.
    //! v0     = [0, 1]
    //! v1     = [2, 3]
    //! result = [1, 3]
    /*!
      \param v0 Vector from which hi elements are alternately selected
      \param v1 Vector from which hi elements are alternately selected
    */
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackHi(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1);
} // namespace rocwmma

#include "vector_util_impl.hpp"

#endif // ROCWMMA_VECTOR_UTIL_HPP
