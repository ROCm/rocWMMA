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
#include "pack.hpp"
#include "permute.hpp"
#include "unpack.hpp"
#include "utils.hpp"

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

    ///
    /// AOS -> SOA : Transform from inline VW to ortho VW
    ///
    template <typename DataT, uint32_t VecSize, uint32_t... Idx>
    ROCWMMA_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v,
                                                            detail::SeqT<Idx...>)
    {
        static_assert(sizeof...(Idx) == VecSize / 2u, "Index count must be half the vector size");
        return VecT<DataT, VecSize / 2u>{get<Idx * 2>(v)...};
    }

    template <typename DataT, uint32_t VecSize, uint32_t... Idx>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v,
                                                           detail::SeqT<Idx...>)
    {
        static_assert(sizeof...(Idx) == VecSize / 2u, "Index count must be half the vector size");
        return VecT<DataT, VecSize / 2u>{get<Idx * 2 + 1>(v)...};
    }

    template <typename DataT, uint32_t VecSize, uint32_t... Idx>
    ROCWMMA_DEVICE constexpr static inline auto
        concat(VecT<DataT, VecSize> const& v0, VecT<DataT, VecSize> const& v1, detail::SeqT<Idx...>)
    {
        static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
        return VecT<DataT, VecSize * 2u>{get<Idx>(v0)..., get<Idx>(v1)...};
    }

    template <typename DataT, uint32_t VecSize, uint32_t... Idx>
    ROCWMMA_DEVICE constexpr static inline auto
        zip(VecT<DataT, VecSize> const& v0, VecT<DataT, VecSize> const& v1, detail::SeqT<Idx...>)
    {
        static_assert(sizeof...(Idx) == VecSize, "Index count must equal the vector size");
        return VecT<DataT, VecSize / 2u>{(get<Idx>(v0), get<Idx>(v1))...};
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v)
    {
        return extractEven(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v)
    {
        return extractOdd(v, detail::Seq<VecSize / 2u>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                       VecT<DataT, VecSize> const& v1)
    {
        return concat(v0, v1, detail::Seq<VecSize>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                    VecT<DataT, VecSize> const& v1)
    {
        return zip(v0, v1, detail::Seq<VecSize>{});
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo2(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Blend::Zip2::exec(
            Packer::exec(extractEven(v)), Dpp::RotateR16<2>::exec(Packer::exec(extractOdd(v)))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo4(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Dpp::template RotateR16<4, 0xF, 0xA>::exec(
            Packer::exec(extractOdd(v)), Packer::exec(extractEven(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo8(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Dpp::template RotateR16<8, 0xF, 0xC>::exec(
            Packer::exec(extractOdd(v)), Packer::exec(extractEven(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi2(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Blend::Zip2::exec(
            Dpp::RotateR16<14>::exec(Packer::exec(extractEven(v))), Packer::exec(extractOdd(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi4(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Dpp::template RotateR16<12, 0xF, 0x5>::exec(
            Packer::exec(extractEven(v)), Packer::exec(extractOdd(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi8(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using Packer   = Pack<DataT, VecSize / 2u>;
        using Unpacker = Unpack<DataT, VecSize / 4u>;
        return Unpacker::exec(Dpp::template RotateR16<8, 0xF, 0x3>::exec(
            Packer::exec(extractEven(v)), Packer::exec(extractOdd(v))));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 8> const& v)
    {
        // Step 1 : Unpack groups of 2
        auto result = concat(unpackLo2(v), unpackHi2(v));

        // Step 2 : Unpack groups of 4
        result = concat(unpackLo4(result), unpackHi4(result));

        // Step 3 : Unpack groups of 8
        result = concat(unpackLo8(result), unpackHi8(result));

        using Packer   = Pack<DataT, 8>;
        using Unpacker = Unpack<DataT, 4>;

        // Step 4 : Swizzle
        return Unpacker::exec(Permute::Gather16<8, 0>::exec(Packer::exec(result)));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
        // // Step 1 : Shift and mix groups of 4
        // {
        //     using DppRotateR16_4_0xF_0xA  = Dpp<DppImpl::Ops::RotateR16<4>, 0xF, 0xA>;
        //     using DppRotateR16_12_0xF_0x5 = Dpp<DppImpl::Ops::RotateR16<12>, 0xF, 0x5>;

        //     auto v0 = get<0>(v);
        //     auto v1 = get<1>(v);
        //     auto v2 = get<2>(v);
        //     auto v3 = get<3>(v);
        //     auto v4 = get<4>(v);
        //     auto v5 = get<5>(v);
        //     auto v6 = get<6>(v);
        //     auto v7 = get<7>(v);

        //     get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v1);
        //     get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v2);
        //     get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v2, v3);
        //     get<4>(v) = DppRotateR16_4_0xF_0xA::exec(v5, v4);
        //     get<5>(v) = DppRotateR16_12_0xF_0x5::exec(v4, v5);
        //     get<6>(v) = DppRotateR16_4_0xF_0xA::exec(v7, v6);
        //     get<7>(v) = DppRotateR16_12_0xF_0x5::exec(v6, v7);
        // }

        // // Step 2: Shift and mix groups of 8
        // {
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;

        //     auto v0 = get<0>(v);
        //     auto v1 = get<1>(v);
        //     auto v2 = get<2>(v);
        //     auto v3 = get<3>(v);
        //     auto v4 = get<4>(v);
        //     auto v5 = get<5>(v);
        //     auto v6 = get<6>(v);
        //     auto v7 = get<7>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v2, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v1);
        //     get<2>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v2);
        //     get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v3);
        //     get<4>(v) = DppRotateR16_8_0xF_0xC::exec(v6, v4);
        //     get<5>(v) = DppRotateR16_8_0xF_0xC::exec(v7, v5);
        //     get<6>(v) = DppRotateR16_8_0xF_0x3::exec(v4, v6);
        //     get<7>(v) = DppRotateR16_8_0xF_0x3::exec(v5, v7);
        // }

        // // Step 3: Rotate latter half of the registers
        // {
        //     using SwzRotateR32_16 = Swizzle<SwizzleImpl::Ops::RotateR32<16>>;
        //     SwzRotateR32_16::exec(get<4>(v));
        //     SwzRotateR32_16::exec(get<5>(v));
        //     SwzRotateR32_16::exec(get<6>(v));
        //     SwzRotateR32_16::exec(get<7>(v));
        // }

        // // Step 4:
        // {
        //     using DppMMove_0x5_0xF = Dpp<DppImpl::Ops::MaskMove, 0x5, 0xF>;

        //     auto v0 = get<0>(v);
        //     auto v1 = get<1>(v);
        //     auto v2 = get<2>(v);
        //     auto v3 = get<3>(v);
        //     auto v4 = get<4>(v);
        //     auto v5 = get<5>(v);
        //     auto v6 = get<6>(v);
        //     auto v7 = get<7>(v);

        //     get<0>(v) = DppMMove_0x5_0xF::exec(v0, v4);
        //     get<1>(v) = DppMMove_0x5_0xF::exec(v1, v5);
        //     get<2>(v) = DppMMove_0x5_0xF::exec(v2, v6);
        //     get<3>(v) = DppMMove_0x5_0xF::exec(v3, v7);
        //     get<4>(v) = DppMMove_0x5_0xF::exec(v4, v0);
        //     get<5>(v) = DppMMove_0x5_0xF::exec(v5, v1);
        //     get<6>(v) = DppMMove_0x5_0xF::exec(v6, v2);
        //     get<7>(v) = DppMMove_0x5_0xF::exec(v7, v3);
        // }

        // // Step 5: Permute
        // {
        //     using Gather32_8_0 = Permute<PermuteImpl::Ops::Gather32<8, 0>>;
        //     using Gather32_8_16 = Permute<PermuteImpl::Ops::Gather32<8, 16>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Gather32_8_0::exec(get<0>(v), threadIdx.x % waveSize);
        //     Gather32_8_0::exec(get<1>(v), threadIdx.x % waveSize);
        //     Gather32_8_0::exec(get<2>(v), threadIdx.x % waveSize);
        //     Gather32_8_0::exec(get<3>(v), threadIdx.x % waveSize);
        //     Gather32_8_16::exec(get<4>(v), threadIdx.x % waveSize);
        //     Gather32_8_16::exec(get<5>(v), threadIdx.x % waveSize);
        //     Gather32_8_16::exec(get<6>(v), threadIdx.x % waveSize);
        //     Gather32_8_16::exec(get<7>(v), threadIdx.x % waveSize);
        // }
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    // SOA -> AOS
    // Transform from ortho VW to inline VW
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 8>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 4>& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 2>& v)
    {
        return 0;
    }

    template <typename DataT>
    __device__ auto aos_soa_16x16_vw4_b32_opt(VecT<DataT, 4>& v)
    {
        // Step 1
        // {

        //     using DppRotateR16_4_0xF_0xA  = Dpp<DppImpl::Ops::RotateR16<4>, 0xF, 0xA>;
        //     using DppRotateR16_12_0xF_0x5 = Dpp<DppImpl::Ops::RotateR16<12>, 0xF, 0x5>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v1);
        //     get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v2);
        //     get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v2, v3);
        // }

        // // Step 2
        // {
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v2, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v1);
        //     get<2>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v2);
        //     get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v3);
        // }

        // // Step 3
        // {
        //     using Gather16_4_0 = Permute<PermuteImpl::Ops::Gather16<4, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Gather16_4_0::exec(v, threadIdx.x % waveSize);
        // }
        return 0;
    }

    template <typename DataT>
    __device__ auto aos_soa_16x8_vw2_b32_opt(VecT<DataT, 2>& v)
    {
        // Step 1
        // {
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v1);
        // }

        // // Step 2
        // {
        //     using Gather16_2_0 = Permute<PermuteImpl::Ops::Gather16<2, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Gather16_2_0::exec(v, threadIdx.x % waveSize);
        // }
        return 0;
    }

    template <typename DataT>
    __device__ auto aos_soa_32x4_vw2_b32_opt(VecT<DataT, 2>& v)
    {
        // Step 1
        // {
        //     using SwzRotateR32_16 = Swizzle<SwizzleImpl::Ops::RotateR32<16>>;
        //     SwzRotateR32_16::exec(get<0>(v));
        // }

        // // Step 2
        // {
        //     using DppMMove_0x5_0xF = Dpp<DppImpl::Ops::MaskMove, 0x5, 0xF>;
        //     using DppMMove_0xA_0xF = Dpp<DppImpl::Ops::MaskMove, 0xA, 0xF>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);

        //     get<0>(v) = DppMMove_0x5_0xF::exec(v1, v0);
        //     get<1>(v) = DppMMove_0xA_0xF::exec(v1, v0);
        // }

        // // Step 3
        // {
        //     using Gather32_2_16 = Permute<PermuteImpl::Ops::Gather32<2, 16>>;
        //     using Gather32_2_0 = Permute<PermuteImpl::Ops::Gather32<2, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Gather32_2_16::exec(get<0>(v), threadIdx.x % waveSize);
        //     Gather32_2_0::exec(get<1>(v), threadIdx.x % waveSize);
        // }
        return 0;
    }

    template <typename DataT>
    __device__ auto aos_soa_32x8_vw4_b32_opt(VecT<DataT, 4>& v)
    {
        // {
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v1);
        //     get<2>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v2);
        //     get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v2, v3);
        // }

        // // Step 1
        // {
        //     using SwzRotateR32_16 = Swizzle<SwizzleImpl::Ops::RotateR32<16>>;
        //     SwzRotateR32_16::exec(get<2>(v));
        //     SwzRotateR32_16::exec(get<3>(v));
        // }

        // // Step 2
        // {
        //     using DppMMove_0x5_0xF = Dpp<DppImpl::Ops::MaskMove, 0x5, 0xF>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppMMove_0x5_0xF::exec(v0, v2);
        //     get<1>(v) = DppMMove_0x5_0xF::exec(v1, v3);
        //     get<2>(v) = DppMMove_0x5_0xF::exec(v2, v0);
        //     get<3>(v) = DppMMove_0x5_0xF::exec(v3, v1);
        // }

        // // Step 3
        // {
        //     using Gather32_4_16 = Permute<PermuteImpl::Ops::Gather32<4, 16>>;
        //     using Gather32_4_0 = Permute<PermuteImpl::Ops::Gather32<4, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Gather32_4_0::exec(get<0>(v), threadIdx.x % waveSize);
        //     Gather32_4_0::exec(get<1>(v), threadIdx.x % waveSize);
        //     Gather32_4_16::exec(get<2>(v), threadIdx.x % waveSize);
        //     Gather32_4_16::exec(get<3>(v), threadIdx.x % waveSize);
        // }
        return 0;
    }

    template <typename DataT>
    __device__ auto soa_aos_16x16_vw4_b32_opt(VecT<DataT, 4>& v)
    {
        // // Step 1
        // {
        //     using Scatter16_4_0 = Permute<PermuteImpl::Ops::Scatter16<4, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Scatter16_4_0::exec(v, threadIdx.x % waveSize);
        // }

        // // Step 2
        // {
        //     using DppRotateR16_4_0xF_0xA  = Dpp<DppImpl::Ops::RotateR16<4>, 0xF, 0xA>;
        //     using DppRotateR16_12_0xF_0x5 = Dpp<DppImpl::Ops::RotateR16<12>, 0xF, 0x5>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppRotateR16_4_0xF_0xA::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_12_0xF_0x5::exec(v0, v1);
        //     get<2>(v) = DppRotateR16_4_0xF_0xA::exec(v3, v2);
        //     get<3>(v) = DppRotateR16_12_0xF_0x5::exec(v2, v3);
        // }

        // // Step 3
        // {
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);
        //     auto const v2 = get<2>(v);
        //     auto const v3 = get<3>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v2, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0xC::exec(v3, v1);
        //     get<2>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v2);
        //     get<3>(v) = DppRotateR16_8_0xF_0x3::exec(v1, v3);
        // }
        return 0;
    }

    template <typename DataT>
    __device__ auto soa_aos_16x8_vw2_b32_opt(VecT<DataT, 2>& v)
    {
        // // Step 1
        // {
        //     using Scatter16_2_0 = Permute<PermuteImpl::Ops::Scatter16<2, 0>>;

        //     constexpr uint32_t waveSize = 64u;
        //     Scatter16_2_0::exec(v, threadIdx.x % waveSize);
        // }

        // // Step 2
        // {
        //     using DppRotateR16_8_0xF_0x3 = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0x3>;
        //     using DppRotateR16_8_0xF_0xC = Dpp<DppImpl::Ops::RotateR16<8>, 0xF, 0xC>;

        //     auto const v0 = get<0>(v);
        //     auto const v1 = get<1>(v);

        //     get<0>(v) = DppRotateR16_8_0xF_0xC::exec(v1, v0);
        //     get<1>(v) = DppRotateR16_8_0xF_0x3::exec(v0, v1);
        // }
        return 0;
    }

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_IMPL_HPP
