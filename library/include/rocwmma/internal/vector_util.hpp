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

#include "blend.hpp"
#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <uint32_t N>
        using Number = detail::integral_constant<int32_t, N>;

        // Can be used to build any vector class of <DataT, VecSize>
        // Either VecT or non_native_vector_vase.
        // Class acts as a static for_each style generator:
        // Incoming functor F will be called with each index + args in sequence.
        // Results of functor calls are used to construct a new vector.
        template <template <typename, uint32_t> class VecT, typename DataT, uint32_t VecSize>
        struct vector_generator_base
        {
            __host__ __device__ constexpr vector_generator() {}

            // F signature: F(Number<Iter>, args...)
            template <class F, uint32_t... Indices, typename... ArgsT>
            __host__ __device__ constexpr auto
                operator()(F f, SeqT<Indices...>, ArgsT&&... args) const
            {
                // Execute incoming functor f with each index, as well as forwarded args.
                // The resulting vector is constructed with the results of each functor call.
                return VecT<DataT, VecSize>{(f(Number<Is>{}, std::forward<ArgsT>(args)...))...};
            }

            template <class F, typename... ArgsT>
            __host__ __device__ constexpr auto operator()(F f, ArgsT&&... args) const
            {
                // Build the number sequence to be expanded above.
                return operator(f, Seq<VecSize>{}, std::forward<ArgsT>(args)...);
            }
        };
    }

    // Specialize for VecT
    template <typename DataT, uint32_t VecSize>
    struct vector_generator : public detail::vector_generator_base<VecT, DataT, VecSize>
    {
    };

    ///////////////////////////////////////////////////////////////////
    ///                 Vector manipulation identities              ///
    ///                                                             ///
    /// Note: performs static unroll                                ///
    ///////////////////////////////////////////////////////////////////

    namespace detail
    {
        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v,
                                                                detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize / 2u,
                          "Index count must be half the vector size");

            if constexpr(sizeof(DataT) == 2u && VecSize >= 4u)
            {
                using PackUtil = PackUtil<DataT>;
                auto p         = PackUtil::paddedPack(v);

                using VecTraits               = VecTraits<decltype(p)>;
                constexpr uint32_t ResultSize = VecTraits::size() / 2;
                using ResultVec               = VecT<typename VecTraits::DataT, ResultSize>;

                ResultVec r;
                for(int i = 0; i < ResultSize; i++)
                {
                    r[i] = Blend::ExtractWordEven(p[i * 2], p[i * 2 + 1]);
                }

                return PackUtil::paddedUnpack(r);
            }
            else if constexpr(sizeof(DataT) == 1u && VecSize >= 8u)
            {
                using PackUtil = PackUtil<DataT>;
                auto p         = PackUtil::paddedPack(v);

                using VecTraits               = VecTraits<decltype(p)>;
                constexpr uint32_t ResultSize = VecTraits::size() / 2;
                using ResultVec               = VecT<typename VecTraits::DataT, ResultSize>;

                ResultVec r;
                for(int i = 0; i < ResultSize; i++)
                {
                    r[i] = Blend::ExtractByteEven(p[i * 2], p[i * 2 + 1]);
                }

                return PackUtil::paddedUnpack(r);
            }
            else
            {
                return VecT<DataT, VecSize / 2u>{get<Idx * 2>(v)...};
            }
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v,
                                                               detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize / 2u,
                          "Index count must be half the vector size");
            return VecT<DataT, VecSize / 2u>{get<Idx * 2 + 1>(v)...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto extractLo(VecT<DataT, VecSize> const& v,
                                                              detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize / 2u,
                          "Index count must be half the vector size");
            return VecT<DataT, VecSize / 2u>{get<Idx>(v)...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto extractHi(VecT<DataT, VecSize> const& v,
                                                              detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize / 2u,
                          "Index count must be half the vector size");
            return VecT<DataT, VecSize / 2u>{get<VecSize / 2 + Idx>(v)...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                           VecT<DataT, VecSize> const& v1,
                                                           detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
            return VecT<DataT, VecSize * 2u>{get<Idx>(v0)..., get<Idx>(v1)...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                        VecT<DataT, VecSize> const& v1,
                                                        detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
            return VecT<DataT, VecSize>{((Idx % 2 == 0) ? get<Idx>(v0) : get<Idx>(v1))...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto unpackLo(VecT<DataT, VecSize> const& v0,
                                                             VecT<DataT, VecSize> const& v1,
                                                             detail::SeqT<Idx...>)
        {
            static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
            return VecT<DataT, VecSize>{
                ((Idx % 2 == 0) ? get<Idx / 2u>(v0) : get<Idx / 2u>(v1))...};
        }

        template <typename DataT, uint32_t VecSize, uint32_t... Idx>
        ROCWMMA_DEVICE constexpr static inline auto unpackHi(VecT<DataT, VecSize> const& v0,
                                                             VecT<DataT, VecSize> const& v1,
                                                             detail::SeqT<Idx...>)
        {
            constexpr auto startIdx = VecSize / 2u;
            static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
            return VecT<DataT, VecSize>{
                ((Idx % 2 == 0) ? get<startIdx + Idx / 2u>(v0) : get<startIdx + Idx / 2u>(v1))...};
        }

    } // namespace detail

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v)
    {
        return detail::extractEven(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractLo(VecT<DataT, VecSize> const& v)
    {
        return detail::extractLo(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractHi(VecT<DataT, VecSize> const& v)
    {
        return detail::extractHi(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v)
    {
        return detail::extractOdd(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                       VecT<DataT, VecSize> const& v1)
    {
        return detail::concat(v0, v1, detail::Seq<VecSize>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                    VecT<DataT, VecSize> const& v1)
    {
        return detail::zip(v0, v1, detail::Seq<VecSize>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackLo(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        return detail::unpackLo(v0, v1, detail::Seq<VecSize>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackHi(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        return detail::unpackHi(v0, v1, detail::Seq<VecSize>{});
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_UTIL_HPP
