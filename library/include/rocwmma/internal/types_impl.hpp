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

    namespace detail
    {
        namespace ArithmeticOp
        {
            struct Add
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs + rhs;
                }
            };
            struct Sub
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs - rhs;
                }
            };
            struct Mult
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs * rhs;
                }
            };
            struct Div
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs / rhs;
                }
            };
            struct Mod
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs % rhs;
                }
            };
            struct Minus
            {
                template <typename TT,
                          typename std::enable_if<std::is_signed<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return -lhs;
                }
            };

        } // namespace ArithmeticOp

        namespace BitwiseOp
        {
            struct And
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs & rhs;
                }
            };

            struct Or
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs | rhs;
                }
            };

            struct Not
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return ~lhs;
                }
            };

            struct Xor
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs ^ rhs;
                }
            };

            struct ShiftR
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

            struct ShiftL
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

        } // namespace BitwiseOp

        namespace LogicalOp
        {
            struct And
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs && rhs;
                }
            };

            struct Or
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs || rhs;
                }
            };

            struct Not
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return !lhs;
                }
            };

        } // namespace LogicalOp

        namespace RelationalOp
        {
            struct Eq
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs == rhs;
                }
            };

            struct Neq
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs != rhs;
                }
            };

            struct Gte
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs >= rhs;
                }
            };

            struct Lte
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs <= rhs;
                }
            };

            struct Gt
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs > rhs;
                }
            };

            struct Lt
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs < rhs;
                }
            };

        } // namespace RelationalOp

        template <class IntT, IntT val>
        struct integral_constant
        {
            static constexpr IntT value = val;
            using value_type            = IntT;
            using type                  = integral_constant;
            constexpr operator value_type() const noexcept
            {
                return value;
            }
            constexpr value_type operator()() const noexcept
            {
                return value;
            }
        };

        template <typename Int, Int... Ints>
        struct integer_sequence
        {
            using value_type = Int;
            constexpr integer_sequence() {}
            static constexpr std::size_t size() noexcept
            {
                return sizeof...(Ints);
            }
        };

        template <std::size_t... Indices>
        using index_sequence = integer_sequence<std::size_t, Indices...>;

        namespace
        {
            // Merge two integer sequences, adding an offset to the right-hand side.
            template <typename Offset, typename Lhs, typename Rhs>
            struct merge;

            template <typename Int, Int Offset, Int... Lhs, Int... Rhs>
            struct merge<integral_constant<Int, Offset>,
                         integer_sequence<Int, Lhs...>,
                         integer_sequence<Int, Rhs...>>
            {
                using type = integer_sequence<Int, Lhs..., (Offset + Rhs)...>;
            };

            template <typename Int, typename N>
            struct log_make_sequence
            {
                using L    = integral_constant<Int, N::value / 2>;
                using R    = integral_constant<Int, N::value - L::value>;
                using type = typename merge<L,
                                            typename log_make_sequence<Int, L>::type,
                                            typename log_make_sequence<Int, R>::type>::type;
            };

            // An empty sequence.
            template <typename Int>
            struct log_make_sequence<Int, integral_constant<Int, 0>>
            {
                using type = integer_sequence<Int>;
            };

            // A single-element sequence.
            template <typename Int>
            struct log_make_sequence<Int, integral_constant<Int, 1>>
            {
                using type = integer_sequence<Int, 0>;
            };
        }

        template <typename Int, Int N>
        using make_integer_sequence =
            typename log_make_sequence<Int, integral_constant<Int, N>>::type;

        template <std::size_t N>
        using make_index_sequence = make_integer_sequence<std::size_t, N>;

        // Helpers for expression expansion, specific to non_native_vector_base
        template <uint32_t... ns>
        using SeqT = integer_sequence<uint32_t, ns...>;
        template <uint32_t Rank>
        using Seq = make_integer_sequence<uint32_t, Rank>;

        // Use with operations that have 2 operands
        template <class BinOp, typename VecT, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline VecT
            binOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return VecT{(BinOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        // Use with operations that have 1 operands
        template <class UnOp, typename VecT, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline VecT unOp(VecT const& lhs,
                                                          SeqT<indices...>) noexcept
        {
            return VecT{(UnOp::exec(lhs.d[indices]))...};
        }

        // Use with relational operations that have 2 operands
        template <class BoolOp, typename VecT, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline VecT
            boolOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            return typename VecT::BoolVecT{(BoolOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        // Used to broadcast a single value to the entire vector.
        template <typename VecT, typename T, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline VecT bCast(T val, SeqT<indices...>) noexcept
        {
            // Indices value ignored, but broadcast for all
            return VecT{((void)indices, val)...};
        }

    } // namespace detail

    // NOTE: The single-valued 'broadcast' constructor can only be used with Rank > 1.
    // Why? The bCast function constructs the vector with <Rank> copies of the input val.
    // When Rank == 1, this would create an endless ctor->bcast->ctor->bcast... loop.
    // As a solution, Rank == 1 should fall into the ctor(Ts... args) for initializer
    // list construction, and NOT bCast initialization.
    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<(std::is_same<U, T>{}) && (Rank > 1)>::type*>
    __HOST_DEVICE__ constexpr non_native_vector_base<T, Rank>::non_native_vector_base(T x_) noexcept
        : non_native_vector_base(detail::template bCast<VecT>(x_, detail::Seq<Rank>{}))
    {
    }

    template <typename T, unsigned int Rank>
    template <typename... Ts,
              typename U,
              typename std::enable_if<(sizeof...(Ts) == Rank)
#if(__cplusplus >= 201703L)
                                      && (std::is_same<U, Ts>{} && ...)
#endif
                                      >::type*>
    __HOST_DEVICE__ constexpr non_native_vector_base<T, Rank>::non_native_vector_base(
        Ts... args) noexcept
        : d{args...}
    {
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ constexpr inline T&
        non_native_vector_base<T, Rank>::operator[](unsigned int idx) noexcept
    {
        return d[idx];
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ constexpr inline T
        non_native_vector_base<T, Rank>::operator[](unsigned int idx) const noexcept
    {
        return d[idx];
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator+=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Add>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator-=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Sub>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator*=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Mult>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator/=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Div>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator%=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Mod>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_signed<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator-() const noexcept -> VecT
    {
        return detail::unOp<detail::ArithmeticOp::Minus>(*this, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator&=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::And>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator|=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::Or>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator~() const noexcept -> VecT
    {
        return detail::unOp<detail::BitwiseOp::Not>(*this, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto non_native_vector_base<T, Rank>::operator^=(const VecT& x_) noexcept
        -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::Xor>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator>>=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::ShiftR>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, typename std::enable_if<std::is_integral<U>{}>::type*>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator<<=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::ShiftL>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator==(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Eq>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator!=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Neq>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator>=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Gte>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator<=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Lte>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator>(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Gt>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    __HOST_DEVICE__ inline auto
        non_native_vector_base<T, Rank>::operator<(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Lt>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator+(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} += y;
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator+(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} += non_native_vector_base<T, n>{y};
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator+(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} += y;
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator-(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} -= y;
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator-(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} -= non_native_vector_base<T, n>{y};
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator-(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} -= y;
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator*(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} *= y;
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator*(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} *= non_native_vector_base<T, n>{y};
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator*(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} *= y;
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator/(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} /= y;
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator/(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} /= non_native_vector_base<T, n>{y};
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator/(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} /= y;
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr bool operator==(const non_native_vector_base<T, n>& x,
                                                     const non_native_vector_base<T, n>& y) noexcept
    {
        return _hip_any_zero(x.data == y.data, n - 1);
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr bool operator==(const non_native_vector_base<T, n>& x,
                                                     U                                   y) noexcept
    {
        return x == non_native_vector_base<T, n>{y};
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr bool operator==(U                                   x,
                                                     const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} == y;
    }

    template <typename T, unsigned int n>
    __HOST_DEVICE__ inline constexpr bool operator!=(const non_native_vector_base<T, n>& x,
                                                     const non_native_vector_base<T, n>& y) noexcept
    {
        return !(x == y);
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr bool operator!=(const non_native_vector_base<T, n>& x,
                                                     U                                   y) noexcept
    {
        return !(x == y);
    }
    template <typename T, unsigned int n, typename U>
    __HOST_DEVICE__ inline constexpr bool operator!=(U                                   x,
                                                     const non_native_vector_base<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator%(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} %= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator%(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} %= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator%(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} %= y;
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator^(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} ^= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator^(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} ^= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator^(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} ^= y;
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator|(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} |= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator|(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} |= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator|(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} |= y;
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator&(const non_native_vector_base<T, n>& x,
                  const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} &= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator&(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} &= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator&(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} &= y;
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator>>(const non_native_vector_base<T, n>& x,
                   const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} >>= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator>>(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} >>= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator>>(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} >>= y;
    }

    template <typename T, unsigned int n, typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator<<(const non_native_vector_base<T, n>& x,
                   const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} <<= y;
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator<<(const non_native_vector_base<T, n>& x, U y) noexcept
    {
        return non_native_vector_base<T, n>{x} <<= non_native_vector_base<T, n>{y};
    }
    template <typename T,
              unsigned int n,
              typename U,
              typename std::enable_if<std::is_arithmetic<U>::value>::type,
              typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__ inline constexpr non_native_vector_base<T, n>
        operator<<(U x, const non_native_vector_base<T, n>& y) noexcept
    {
        return non_native_vector_base<T, n>{x} <<= y;
    }

} // namespace rocwmma

#endif // ROCWMMA_TYPES_IMPL_HPP
