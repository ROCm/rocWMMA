/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_VECTOR_ITERATOR_HPP
#define ROCWMMA_VECTOR_ITERATOR_HPP

#include "vector.hpp"

inline constexpr unsigned int next_pot(unsigned int x)
{
    // Precondition: x > 1.
    return x > 1 ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}
namespace rocwmma
{

    // Vector iterator class: handles for const and non-const vectors
    template <class VecT, uint32_t SubVecSize = 1>
    struct VectorIterator;

    template <typename DataT, uint32_t Rank, uint32_t SubVecSize>
    struct VectorIterator<HIP_vector_type<DataT, Rank>, SubVecSize>
    {
        template <typename VDataT, uint32_t VRank>
        using VecT = HIP_vector_type<VDataT, VRank>;

        using RefVecT = HIP_vector_type<DataT, Rank>;
        using ItVecT  = HIP_vector_type<DataT, SubVecSize>;

        struct iterator
        {
            RefVecT const&   mRef;
            mutable uint32_t mIdx;

            struct Traits
            {
                enum : int32_t
                {
                    Range = Rank / SubVecSize
                };
            };

            static_assert(Rank % SubVecSize == 0, "VecSize not iterable by SubVecSize");
            static_assert(sizeof(RefVecT) == sizeof(typename RefVecT::Native_vec_),
                          "Cannot alias subvector");
            static_assert(sizeof(ItVecT) == sizeof(typename ItVecT::Native_vec_),
                          "Cannot alias subvector");
            static_assert(sizeof(RefVecT) == sizeof(ItVecT) * Traits::Range,
                          "Cannot alias subvector");

            ROCWMMA_HOST_DEVICE constexpr iterator() noexcept = delete;

            ROCWMMA_HOST_DEVICE constexpr iterator(RefVecT const& ref, uint32_t idx = 0) noexcept
                : mRef(ref)
                , mIdx(idx)
            {
            }

            ROCWMMA_HOST_DEVICE ~iterator() = default;

            ROCWMMA_HOST_DEVICE inline ItVecT const& operator*() const
            {
                // Cast as array of sub-vectors
                return reinterpret_cast<ItVecT const*>(&mRef)[mIdx];
            }

            ROCWMMA_HOST_DEVICE inline ItVecT& operator*()
            {
                // Cast as array of sub-vectors
                return reinterpret_cast<ItVecT*>(&const_cast<RefVecT&>(mRef))[mIdx];
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator++() const
            {
                mIdx++;
                return *this;
            }
            ROCWMMA_HOST_DEVICE inline iterator operator++(int) const
            {
                auto retval = *this;
                ++mIdx;
                return retval;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator--() const
            {
                mIdx--;
                return *this;
            }
            ROCWMMA_HOST_DEVICE inline iterator operator--(int) const
            {
                auto retval = *this;
                --mIdx;
                return retval;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator+=(int i) const
            {
                mIdx += i;
                return *this;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator-=(int i) const
            {
                mIdx -= i;
                return *this;
            }

            ROCWMMA_HOST_DEVICE inline iterator operator+(int i) const
            {
                auto retval = *this;
                return retval += i;
            }

            ROCWMMA_HOST_DEVICE inline iterator operator-(int i) const
            {
                auto retval = *this;
                return retval -= i;
            }

            ROCWMMA_HOST_DEVICE inline bool operator==(iterator const& other) const
            {
                return (&mRef == &other.mRef) && (mIdx == other.mIdx);
            }
            ROCWMMA_HOST_DEVICE inline bool operator!=(iterator const& other) const
            {
                return !(*this == other);
            }

            //__device__ inline Iterator<SubVecSize, IsConst> next() const;
            // __device__ inline Iterator<SubVecSize, IsConst> prev() const;
            ROCWMMA_HOST_DEVICE inline uint32_t index() const
            {
                return mIdx;
            }
            ROCWMMA_HOST_DEVICE inline bool valid() const
            {
                return (mIdx >= 0) && (mIdx < Traits::Range);
            }
            // __device__ bool                                 valid() const;

            ROCWMMA_HOST_DEVICE constexpr static inline int32_t range()
            {
                return Traits::Range;
            }
            // __device__ constexpr static inline bool    isConst();
        };

        ROCWMMA_HOST_DEVICE
        constexpr VectorIterator(RefVecT const& refVec) noexcept
            : mRef(refVec)
        {
        }

        ROCWMMA_HOST_DEVICE
        ~VectorIterator() = default;

        ROCWMMA_HOST_DEVICE
        inline iterator it(uint32_t startIdx = 0)
        {
            return iterator(mRef, startIdx);
        }

        ROCWMMA_HOST_DEVICE
        inline iterator begin()
        {
            return iterator(mRef, 0u);
        }

        ROCWMMA_HOST_DEVICE
        inline iterator end()
        {
            return iterator(mRef, Rank / SubVecSize);
        }

        RefVecT const& mRef;
    };

    template <uint32_t SubVecSize = 1, typename VecT = void>
    constexpr auto makeVectorIterator(VecT const& vec)
    {
        return VectorIterator<VecT, SubVecSize>{vec};
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_ITERATOR_HPP
