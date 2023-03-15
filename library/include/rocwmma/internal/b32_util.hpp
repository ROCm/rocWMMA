/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
#ifndef B32_UTIL_HPP
#define B32_UTIL_HPP

#include "transforms.hpp"

#include "dpp.hpp"
#include "io_traits.hpp"
#include "permute.hpp"

namespace rocwmma
{
    template <typename DataT>
    struct b32Util
    {
        using PackTraits = detail::PackTraits<DataT>;

        using ExtractedT = typename PackTraits::PackedT;
        using EmplacedT  = typename PackTraits::UnpackedT;

        using FootSpace = union
        {
            ExtractedT extracted;
            EmplacedT  emplaced[PackTraits::PackRatio];
        };

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto extract(VecT<EmplacedT, VecSize> const& v)
        {
            VecT<ExtractedT, VecSize> result;
            auto const                rIt = makeVectorIterator(v).begin();
            auto                      wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(ExtractedT) / sizeof(EmplacedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.emplaced[SelectIdx] = get<0>(*rIt);
                get<0>(*wIt)          = a.extracted;
            }
            return result;
        }

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto emplace(VecT<ExtractedT, VecSize> const& v)
        {
            VecT<EmplacedT, VecSize> result;
            auto const               rIt = makeVectorIterator(v).begin();
            auto                     wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(ExtractedT) / sizeof(EmplacedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.extracted  = get<0>(*rIt);
                get<0>(*wIt) = a.emplaced[SelectIdx];
            }
            return result;
        }

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto emplace(VecT<EmplacedT, VecSize>&        dst,
                                       VecT<ExtractedT, VecSize> const& v)
        {
            auto const rIt = makeVectorIterator(v).begin();
            auto       wIt = makeVectorIterator(dst).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(EmplacedT) / sizeof(ExtractedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.extracted  = get<0>(*rIt);
                get<0>(*wIt) = a.emplaced[SelectIdx];
            }
            return dst;
        }
    };

    template <>
    struct b32Util<float64_t>
    {
        using ExtractedT = float32_t;
        using EmplacedT  = float64_t;

        using FootSpace = union
        {
            ExtractedT extracted[2];
            EmplacedT  emplaced;
        };

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto extract(VecT<EmplacedT, VecSize> const& v)
        {
            VecT<ExtractedT, VecSize> result;
            auto const                rIt = makeVectorIterator(v).begin();
            auto                      wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(EmplacedT) / sizeof(ExtractedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.emplaced   = get<0>(*rIt);
                get<0>(*wIt) = a.extracted[SelectIdx];
            }
            return result;
        }

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto emplace(VecT<ExtractedT, VecSize> const& v)
        {
            VecT<EmplacedT, VecSize> result;
            auto const               rIt = makeVectorIterator(v).begin();
            auto                     wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(EmplacedT) / sizeof(ExtractedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.extracted[SelectIdx] = get<0>(*rIt);
                get<0>(*wIt)           = a.emplaced;
            }
            return result;
        }

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto emplace(VecT<EmplacedT, VecSize>&        dst,
                                       VecT<ExtractedT, VecSize> const& v)
        {
            auto const rIt = makeVectorIterator(v).begin();
            auto       wIt = makeVectorIterator(dst).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(EmplacedT) / sizeof(ExtractedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.emplaced             = get<0>(*wIt);
                a.extracted[SelectIdx] = get<0>(*rIt);
                get<0>(*wIt)           = a.emplaced;
            }
            return dst;
        }
    };

} // namespace rocwmma

#endif // B32_UTIL_HPP
