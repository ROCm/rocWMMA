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
#ifndef ROCWMMA_TRANSFORMS_IMPL_HPP
#define ROCWMMA_TRANSFORMS_IMPL_HPP

#include "transforms.hpp"

#include "dpp.hpp"
#include "io_traits.hpp"
#include "permute.hpp"

namespace rocwmma
{
    template <typename DataT>
    struct Lufi
    {
        using PackTraits = detail::PackTraits<DataT>;

        using InflatedT = typename PackTraits::PackedT;
        using DeflatedT = typename PackTraits::UnpackedT;

        using FootSpace = union
        {
            InflatedT inflated;
            DeflatedT deflated[PackTraits::PackRatio];
        };

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto inflate(VecT<DeflatedT, VecSize> const& v)
        {
            VecT<InflatedT, VecSize> result;
            auto const               rIt = makeVectorIterator(v).begin();
            auto                     wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(InflatedT) / sizeof(DeflatedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.deflated[SelectIdx] = get<0>(*rIt);
                get<0>(*wIt)          = a.inflated;
            }
            return result;
        }

        template <uint32_t VecSize, uint32_t SelectIdx = 0u>
        __device__ static auto deflate(VecT<InflatedT, VecSize> const& v)
        {
            VecT<DeflatedT, VecSize> result;
            auto const               rIt = makeVectorIterator(v).begin();
            auto                     wIt = makeVectorIterator(result).begin();

            static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                          "Unexpected iterator range mismatch");

            static_assert(SelectIdx < (sizeof(InflatedT) / sizeof(DeflatedT)),
                          "Invalid index selection");

            for(uint32_t i = 0u; i < decltype(rIt)::range(); i++, rIt++, wIt++)
            {
                FootSpace a;
                a.inflated   = get<0>(*rIt);
                get<0>(*wIt) = a.deflated[SelectIdx];
            }
            return result;
        }
    };

    template <typename DataT>
    __device__ void aos_soa_16x32_vw8_b32_opt(VecT<DataT, 8>& v)
    {
        // Step 1 : Shift and mix groups of 2
        {
            // For this step, easier to reinterpret as uint32_t.
            // Shift + mix groups of 2
            auto& vv = reinterpret_cast<VecT<uint32_t, 8>&>(v);

            using DppRotateR16_2_0xF_0xF  = Dpp<DppOps::RotateR16<2>, 0xF, 0xF>;
            using DppRotateR16_14_0xF_0xF = Dpp<DppOps::RotateR16<14>, 0xF, 0xF>;

            auto v0 = DppRotateR16_14_0xF_0xF::exec(get<0>(vv));
            auto v1 = DppRotateR16_2_0xF_0xF::exec(get<1>(vv));
            auto v2 = DppRotateR16_14_0xF_0xF::exec(get<2>(vv));
            auto v3 = DppRotateR16_2_0xF_0xF::exec(get<3>(vv));
            auto v4 = DppRotateR16_14_0xF_0xF::exec(get<4>(vv));
            auto v5 = DppRotateR16_2_0xF_0xF::exec(get<5>(vv));
            auto v6 = DppRotateR16_14_0xF_0xF::exec(get<6>(vv));
            auto v7 = DppRotateR16_2_0xF_0xF::exec(get<7>(vv));

            uint32_t mask = ((threadIdx.x >> 1) & 0x1) * LsbMask<32>::value;
            get<0>(vv)    = (v1 & mask) | (get<0>(vv) & ~mask);
            get<1>(vv)    = (get<1>(vv) & mask) | (v0 & ~mask);
            get<2>(vv)    = (v3 & mask) | (get<2>(vv) & ~mask);
            get<3>(vv)    = (get<3>(vv) & mask) | (v2 & ~mask);
            get<4>(vv)    = (v5 & mask) | (get<4>(vv) & ~mask);
            get<5>(vv)    = (get<5>(vv) & mask) | (v4 & ~mask);
            get<6>(vv)    = (v7 & mask) | (get<6>(vv) & ~mask);
            get<7>(vv)    = (get<7>(vv) & mask) | (v6 & ~mask);
        }

        // Step 2: Shift and mix groups of 4
        {
            using DppRotateR16_4_0xF_0xA  = Dpp<DppOps::RotateR16<4>, 0xF, 0xA>;
            using DppRotateR16_12_0xF_0x5 = Dpp<DppOps::RotateR16<12>, 0xF, 0x5>;

            auto v0 = get<0>(v);
            auto v1 = get<1>(v);
            auto v2 = get<2>(v);
            auto v3 = get<3>(v);
            auto v4 = get<4>(v);
            auto v5 = get<5>(v);
            auto v6 = get<6>(v);
            auto v7 = get<7>(v);

            get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v2, v0);
            get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v2);
            get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v1);
            get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v1, v3);
            get<4>(v) = DppRotateR16_4_0xF_0xA::exec(v6, v4);
            get<5>(v) = DppRotateR16_12_0xF_0x5::exec(v4, v6);
            get<6>(v) = DppRotateR16_4_0xF_0xA::exec(v7, v5);
            get<7>(v) = DppRotateR16_12_0xF_0x5::exec(v5, v7);
        }

        // Step 3: Shift and mix groups of 8
        {
            using DppRotateR16_8_0xF_0xC = Dpp<DppOps::RotateR16<8>, 0xF, 0xC>;
            using DppRotateR16_8_0xF_0x3 = Dpp<DppOps::RotateR16<8>, 0xF, 0x3>;

            auto v0 = get<0>(v);
            auto v1 = get<1>(v);
            auto v2 = get<2>(v);
            auto v3 = get<3>(v);
            auto v4 = get<4>(v);
            auto v5 = get<5>(v);
            auto v6 = get<6>(v);
            auto v7 = get<7>(v);

            get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v4, v0);
            get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v6, v2);
            get<2>(v) = DppRotateR16_8_0xF_0xC::exec(v5, v1);
            get<3>(v) = DppRotateR16_8_0xF_0xC::exec(v7, v3);
            get<4>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v4);
            get<5>(v) = DppRotateR16_8_0xF_0x3::exec(v2, v6);
            get<6>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v5);
            get<7>(v) = DppRotateR16_8_0xF_0x3::exec(v3, v7);
        }

        // Step 4 : Swizzle
        {
            using Gather16_8_0 = Permute<PermuteOps::Gather16<8, 0>>;

            constexpr uint32_t waveSize = 64u;
            Gather16_8_0::exec(v, threadIdx.x % waveSize);
        }
    }

    template <typename DataT>
    __device__ void aos_soa_16x16_vw4_b32_opt(VecT<DataT, 4>& v)
    {
        // Step 1
        {

            using DppRotateR16_4_0xF_0xA  = Dpp<DppOps::RotateR16<4>, 0xF, 0xA>;
            using DppRotateR16_12_0xF_0x5 = Dpp<DppOps::RotateR16<12>, 0xF, 0x5>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);
            auto const v2 = get<2>(v);
            auto const v3 = get<3>(v);

            get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v1, v0);
            get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v1);
            get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v2);
            get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v2, v3);
        }

        // Step 2
        {
            using DppRotateR16_8_0xF_0xC = Dpp<DppOps::RotateR16<8>, 0xF, 0xC>;
            using DppRotateR16_8_0xF_0x3 = Dpp<DppOps::RotateR16<8>, 0xF, 0x3>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);
            auto const v2 = get<2>(v);
            auto const v3 = get<3>(v);

            get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v2, v0);
            get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v1);
            get<2>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v2);
            get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v3);
        }

        // Step 3
        {
            using Gather16_4_0 = Permute<PermuteOps::Gather16<4, 0>>;

            constexpr uint32_t waveSize = 64u;
            Gather16_4_0::exec(v, threadIdx.x % waveSize);
        }
    }

    template <typename DataT>
    __device__ void aos_soa_16x8_vw2_b32_opt(VecT<DataT, 2>& v)
    {
        // Step 1
        {
            using DppRotateR16_8_0xF_0x3 = Dpp<DppOps::RotateR16<8>, 0xF, 0x3>;
            using DppRotateR16_8_0xF_0xC = Dpp<DppOps::RotateR16<8>, 0xF, 0xC>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);

            get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v1, v0);
            get<1>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v1);
        }

        // Step 2
        {
            using Gather16_2_0 = Permute<PermuteOps::Gather16<2, 0>>;

            constexpr uint32_t waveSize = 64u;
            Gather16_2_0::exec(v, threadIdx.x % waveSize);
        }
    }

    template <typename DataT>
    __device__ void soa_aos_16x16_vw4_b32_opt(VecT<DataT, 4>& v)
    {
        // Step 1
        {
            using Scatter16_4_0 = Permute<PermuteOps::Scatter16<4, 0>>;

            constexpr uint32_t waveSize = 64u;
            Scatter16_4_0::exec(v, threadIdx.x % waveSize);
        }

        // Step 2
        {
            using DppRotateR16_4_0xF_0xA  = Dpp<DppOps::RotateR16<4>, 0xF, 0xA>;
            using DppRotateR16_12_0xF_0x5 = Dpp<DppOps::RotateR16<12>, 0xF, 0x5>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);
            auto const v2 = get<2>(v);
            auto const v3 = get<3>(v);

            get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v1, v0);
            get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v1);
            get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v2);
            get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v2, v3);
        }

        // Step 3
        {
            using DppRotateR16_8_0xF_0xC = Dpp<DppOps::RotateR16<8>, 0xF, 0xC>;
            using DppRotateR16_8_0xF_0x3 = Dpp<DppOps::RotateR16<8>, 0xF, 0x3>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);
            auto const v2 = get<2>(v);
            auto const v3 = get<3>(v);

            get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v2, v0);
            get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v1);
            get<2>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v2);
            get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v3);
        }
    }

    template <typename DataT>
    __device__ void soa_aos_16x8_vw2_b32_opt(VecT<DataT, 2>& v)
    {
        // Step 1
        {
            using Scatter16_2_0 = Permute<PermuteOps::Scatter16<2, 0>>;

            constexpr uint32_t waveSize = 64u;
            Scatter16_2_0::exec(v, threadIdx.x % waveSize);
        }

        // Step 2
        {
            using DppRotateR16_8_0xF_0x3 = Dpp<DppOps::RotateR16<8>, 0xF, 0x3>;
            using DppRotateR16_8_0xF_0xC = Dpp<DppOps::RotateR16<8>, 0xF, 0xC>;

            auto const v0 = get<0>(v);
            auto const v1 = get<1>(v);

            get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v1, v0);
            get<1>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v1);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_IMPL_HPP
