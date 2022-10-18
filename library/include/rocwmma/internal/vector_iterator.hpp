#ifndef ROCWMMA_VECTOR_ITERATOR_HPP
#define ROCWMMA_VECTOR_ITERATOR_HPP

#include "vector.hpp"
#include <iostream>

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
            RefVecT const& mRef;
            uint32_t       mIdx;

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

            __HOST_DEVICE__ constexpr iterator(RefVecT const& ref, uint32_t idx = 0) noexcept
                : mRef(ref)
                , mIdx(idx)
            {
            }
            __HOST_DEVICE__ ~iterator() = default;

            __HOST_DEVICE__ inline ItVecT const& operator*() const
            {
                return *reinterpret_cast<ItVecT const*>((&mRef.data) + (mIdx * SubVecSize));
            }

            __HOST_DEVICE__ inline ItVecT& operator*()
            {
                return *reinterpret_cast<ItVecT*>((&const_cast<RefVecT&>(mRef).data)
                                                  + (mIdx * SubVecSize));
            }

            __HOST_DEVICE__ inline iterator& operator++()
            {
                mIdx++;
                return *this;
            }
            __HOST_DEVICE__ inline iterator operator++(int)
            {
                auto retval = *this;
                ++mIdx;
                return retval;
            }

            __HOST_DEVICE__ inline iterator& operator+=(int i)
            {
                mIdx += i;
                return *this;
            }

            __HOST_DEVICE__ inline iterator& operator--()
            {
                mIdx--;
                return *this;
            }
            __HOST_DEVICE__ inline iterator operator--(int)
            {
                auto retval = *this;
                --mIdx;
                return retval;
            }
            __HOST_DEVICE__ inline iterator& operator-=(int i)
            {
                mIdx -= i;
                return *this;
            }

            __HOST_DEVICE__ inline bool operator==(iterator const& other)
            {
                return (&mRef == &other.mRef) && (mIdx == other.mIdx);
            }
            __HOST_DEVICE__ inline bool operator!=(iterator const& other)
            {
                return !(*this == other);
            }

            //__device__ inline Iterator<SubVecSize, IsConst> next() const;
            // __device__ inline Iterator<SubVecSize, IsConst> prev() const;
            // __device__ inline int32_t                       index() const;
            // __device__ bool                                 valid() const;

            __device__ constexpr static inline int32_t range();
            // __device__ constexpr static inline bool    isConst();
        };

        __HOST_DEVICE__
        constexpr VectorIterator(RefVecT const& refVec) noexcept
            : mRef(refVec)
        {
        }

        __HOST_DEVICE__
        ~VectorIterator() = default;

        __HOST_DEVICE__
        inline iterator it(uint32_t startIdx = 0)
        {
            return iterator(mRef, startIdx);
        }

        __HOST_DEVICE__
        inline iterator begin()
        {
            return iterator(mRef, 0u);
        }

        __HOST_DEVICE__
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

    inline void doStuff()
    {
        HIP_vector_type<rocwmma::bfloat16_t, 4> yubnub{rocwmma::bfloat16_t(5.0f)};
        for(auto const l : makeVectorIterator(yubnub))
        {
            std::cout << (l.data[0]) << ' '; // 3 4 5
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_ITERATOR_HPP
