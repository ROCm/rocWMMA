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
#ifndef ROCWMMA_TYPES_IMPL_HPP
#define ROCWMMA_TYPES_IMPL_HPP

#include "types.hpp"

namespace rocwmma
{

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ constexpr VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::Iterator(
        ParentT& parent, uint32_t startIndex /*= 0*/)
        : mIndex(startIndex)
        , mParent(parent)
    {
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ constexpr inline int32_t VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::range()
    {
        return Traits::Range;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline int32_t VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::index() const
    {
        return mIndex;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator*() const ->
        typename Traits::ItVecT&
    {
        return *reinterpret_cast<typename Traits::ItVecT const*>(&(mParent[mIndex * SubVecSize]));
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator*() ->
        typename Traits::ItVecT&
    {
        return *reinterpret_cast<typename Traits::ItVecT*>(&(mParent[mIndex * SubVecSize]));
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator++(int)
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex++;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator++()
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex++;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator+=(int i)
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex += i;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator--()
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex--;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator--(int)
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex--;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::operator-=(int i)
        -> Iterator<SubVecSize, IsConst>&
    {
        mIndex -= i;
        return *this;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::next() const
        -> Iterator<SubVecSize, IsConst>
    {
        return Iterator<SubVecSize, IsConst>(mParent, mIndex + 1);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ inline auto VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::prev() const
        -> Iterator<SubVecSize, IsConst>
    {
        return Iterator<SubVecSize, IsConst>(mParent, mIndex - 1);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize, bool IsConst>
    __device__ bool VecT<T, VecSize>::Iterator<SubVecSize, IsConst>::valid() const
    {
        return (mIndex >= 0) && (mIndex < Traits::Range);
    }

    template <typename T, uint32_t VecSize>
    __device__ inline VecT<T, VecSize>::VecT(VecT const& other)
    {
        v = other.v;
    }

    template <typename T, uint32_t VecSize>
    __device__ inline VecT<T, VecSize>::VecT(StorageT const& other)
    {
        v = other;
    }

    template <typename T, uint32_t VecSize>
    __device__ VecT<T, VecSize>::VecT(StorageT&& other)
    {
        v = std::move(other);
    }

    template <typename T, uint32_t VecSize>
    __device__ auto VecT<T, VecSize>::operator[](uint32_t index) -> DataT&
    {
        return e[index];
    }

    template <typename T, uint32_t VecSize>
    __device__ auto VecT<T, VecSize>::operator*() -> StorageT&
    {
        return v;
    }

    template <typename T, uint32_t VecSize>
    __device__ auto VecT<T, VecSize>::operator[](uint32_t index) const -> DataT const&
    {
        return e[index];
    }

    template <typename T, uint32_t VecSize>
    __device__ auto VecT<T, VecSize>::operator*() const -> StorageT const&
    {
        return v;
    }

    template <typename T, uint32_t VecSize>
    __device__ constexpr inline uint32_t VecT<T, VecSize>::size()
    {
        return VecSize;
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::begin() -> iterator<SubVecSize>
    {
        return iterator<SubVecSize>(*this);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::end() -> iterator<SubVecSize>
    {
        return iterator<SubVecSize>(*this, iterator<SubVecSize>::range());
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::it(uint32_t startIndex /*= 0*/) -> iterator<SubVecSize>
    {
        return iterator<SubVecSize>(*this, startIndex);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::begin() const -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::end() const -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this, const_iterator<SubVecSize>::range());
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::it(uint32_t startIndex /*= 0*/) const
        -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this, startIndex);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::cbegin() const -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this);
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::cend() const -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this, const_iterator<SubVecSize>::range());
    }

    template <typename T, uint32_t VecSize>
    template <uint32_t SubVecSize /*= 1*/>
    __device__ inline auto VecT<T, VecSize>::cit(uint32_t startIndex /*= 0*/) const
        -> const_iterator<SubVecSize>
    {
        return const_iterator<SubVecSize>(*this, startIndex);
    }

} // namespace rocwmma

#endif // ROCWMMA_TYPES_IMPL_HPP
