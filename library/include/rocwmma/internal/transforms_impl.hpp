/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "pack_util.hpp"
#include "permute.hpp"
#include "utils.hpp"
#include "vector_util.hpp"

namespace rocwmma
{
    ///
    /// AOS -> SOA : Transform from inline VW to ortho VW
    ///

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo2(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Blend::Zip2::exec(PackUtil::paddedPack(extractEven(v)),
                              Dpp::RotateR16<2>::exec(PackUtil::paddedPack(extractOdd(v)))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo4(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Dpp::template RotateR16<4, 0xF, 0xA>::exec(PackUtil::paddedPack(extractOdd(v)),
                                                       PackUtil::paddedPack(extractEven(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLo8(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Dpp::template RotateR16<8, 0xF, 0xC>::exec(PackUtil::paddedPack(extractOdd(v)),
                                                       PackUtil::paddedPack(extractEven(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi2(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Blend::Zip2::exec(Dpp::RotateR16<14>::exec(PackUtil::paddedPack(extractEven(v))),
                              PackUtil::paddedPack(extractOdd(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi4(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Dpp::template RotateR16<12, 0xF, 0x5>::exec(PackUtil::paddedPack(extractEven(v)),
                                                        PackUtil::paddedPack(extractOdd(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackHi8(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        return PackUtil::template paddedUnpack<VecSize / 2>(
            Dpp::template RotateR16<8, 0xF, 0x3>::exec(PackUtil::paddedPack(extractEven(v)),
                                                       PackUtil::paddedPack(extractOdd(v))));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLoHi2(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        auto evens = PackUtil::paddedPack(extractEven(v));
        auto odds  = PackUtil::paddedPack(extractOdd(v));
        auto lo    = Blend::Zip2::exec(evens, Dpp::RotateR16<2>::exec(odds));
        auto hi    = Blend::Zip2::exec(Dpp::RotateR16<14>::exec(evens), odds);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLoHi4(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        auto evens = PackUtil::paddedPack(extractEven(v));
        auto odds  = PackUtil::paddedPack(extractOdd(v));
        auto lo    = Dpp::template RotateR16<4, 0xF, 0xA>::exec(odds, evens);
        auto hi    = Dpp::template RotateR16<12, 0xF, 0x5>::exec(evens, odds);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLoHi8(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        auto evens = PackUtil::paddedPack(extractEven(v));
        auto odds  = PackUtil::paddedPack(extractOdd(v));
        auto lo    = Dpp::template RotateR16<8, 0xF, 0xC>::exec(odds, evens);
        auto hi    = Dpp::template RotateR16<8, 0xF, 0x3>::exec(evens, odds);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLoHi16(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        auto lo     = PackUtil::paddedPack(extractEven(v));
        auto hi     = PackUtil::paddedPack(extractOdd(v));
        auto rot_lo = Swizzle::RotateR32<16>::exec(lo);
        auto rot_hi = Swizzle::RotateR32<16>::exec(hi);
        lo          = Blend::Zip16::exec(lo, rot_hi);
        hi          = Blend::Zip16::exec(rot_lo, hi);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    // TODO: Wave64 only?
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto unpackLoHi32(VecT<DataT, VecSize> const& v)
    {
        static_assert(VecSize % 2 == 0, "VecSize must be a multiple of 2");
        using PackUtil = PackUtil<DataT>;

        auto lo = PackUtil::paddedPack(extractEven(v));
        auto hi = PackUtil::paddedPack(extractOdd(v));

        // TODO: label as rotateR64 for consistency?
        auto rot_lo = Permute::RotateWaveR<32>::exec(lo);
        auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
        lo          = Blend::Zip32::exec(lo, rot_hi);
        hi          = Blend::Zip32::exec(rot_lo, hi);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 8> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        // Step 1 : Unpack groups of 2
        auto result = unpackLoHi2(v);

        // Step 2 : Unpack groups of 4
        result = unpackLoHi4(result);

        // Step 3 : Unpack groups of 8
        result = unpackLoHi8(result);

        // Step 4 : Gather
        return PackUtil::template paddedUnpack<8>(
            Permute::Gather16<8, 0>::exec(PackUtil::paddedPack(result)));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 4> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        auto result = unpackLoHi4(v);
        result      = unpackLoHi8(result);
        return PackUtil::template paddedUnpack<4>(
            Permute::Gather16<4, 0>::exec(PackUtil::paddedPack(result)));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 8> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        // Step 1 : Unpack groups of 4
        auto result = unpackLoHi4(v);

        // Step 2 : Unpack groups of 8
        result = unpackLoHi8(result);

        // Step 3 : Unpack groups of 16
        // In order to save some operations, we can
        // rotate the odds components only and make up the
        // offset later in gather.
        auto evens = PackUtil::paddedPack(extractEven(result));
        auto odds  = PackUtil::paddedPack(extractOdd(result));

        auto rot = Swizzle::RotateR32<16>::exec(odds);
        auto lo  = Blend::Zip16::exec(evens, rot);
        auto hi  = Blend::Zip16::exec(rot, evens);

        // Step 4 : Gather
        // Note the offset of 16 in hi
        lo = Permute::Gather32<8, 0>::exec(lo);
        hi = Permute::Gather32<8, 16>::exec(hi);

        return PackUtil::template paddedUnpack<8>(concat(lo, hi));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 4> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        auto result = unpackLoHi8(v);

        // modified unpackLohi16
        {
            auto lo     = PackUtil::paddedPack(extractEven(result));
            auto hi     = PackUtil::paddedPack(extractOdd(result));
            auto rot_hi = Swizzle::RotateR32<16>::exec(hi);
            hi          = Blend::Zip16::exec(rot_hi, lo);
            lo          = Blend::Zip16::exec(lo, rot_hi);
            result      = concat(PackUtil::template paddedUnpack<2u>(lo),
                            PackUtil::template paddedUnpack<2u>(hi));
        }

        auto hi = Permute::Gather32<4, 16>::exec(PackUtil::paddedPack(extractHi(result)));
        auto lo = Permute::Gather32<4, 0>::exec(PackUtil::paddedPack(extractLo(result)));

        return concat(PackUtil::template paddedUnpack<2u>(lo),
                      PackUtil::template paddedUnpack<2u>(hi));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 8> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        // Step 1 : Unpack groups of 8
        auto result = unpackLoHi8(v);

        // Step 2 : Unpack groups of 16
        result = unpackLoHi16(result);

        // Step 3 : Unpack groups of 32
        // In order to save some operations, we can
        // rotate the odds components only and make up the
        // offset later in gather.
        auto lo = PackUtil::paddedPack(extractEven(result));
        auto hi = PackUtil::paddedPack(extractOdd(result));

        // TODO: label as rotateR64 for consistency?
        auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
        hi          = Blend::Zip32::exec(rot_hi, lo);
        lo          = Blend::Zip32::exec(lo, rot_hi);

        // Step 4 : Gather
        // Note the offset of 32 in hi
        lo = Permute::GatherWave<8, 0>::exec(lo);
        hi = Permute::GatherWave<8, 32>::exec(hi);

        return PackUtil::template paddedUnpack<8>(concat(lo, hi));
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <uint32_t BlockDim, uint32_t VectorWidth>
    struct AosToSoa;

    template <>
    struct AosToSoa<16, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Unpack groups of 2
            auto result = unpackLoHi2(v);

            // Step 2 : Unpack groups of 4
            result = unpackLoHi4(result);

            // Step 3 : Unpack groups of 8
            result = unpackLoHi8(result);

            // Step 4 : Gather
            return PackUtil::template paddedUnpack<VecSize>(
                Permute::Gather16<VW, 0>::exec(PackUtil::paddedPack(result)));
        }
    };

    template <>
    struct AosToSoa<32, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Unpack groups of 4
            auto result = unpackLoHi4(v);

            // Step 2 : Unpack groups of 8
            result = unpackLoHi8(result);

            // Step 3 : Unpack groups of 16
            // In order to save some operations, we can
            // rotate the odds components only and make up the
            // offset later in gather.
            auto evens = PackUtil::paddedPack(extractEven(result));
            auto odds  = PackUtil::paddedPack(extractOdd(result));

            auto rot = Swizzle::RotateR32<16>::exec(odds);
            auto lo  = Blend::Zip16::exec(evens, rot);
            auto hi  = Blend::Zip16::exec(rot, evens);

            // Step 4 : Gather
            // Note the offset of 16 in hi
            lo = Permute::Gather32<VW, 0>::exec(lo);
            hi = Permute::Gather32<VW, 16>::exec(hi);

            return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
        }
    };

    template <>
    struct AosToSoa<64, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Unpack groups of 8
            auto result = unpackLoHi8(v);

            // Step 2 : Unpack groups of 16
            result = unpackLoHi16(result);

            // Step 3 : Unpack groups of 32
            // In order to save some operations, we can
            // rotate the odds components only and make up the
            // offset later in gather.
            auto lo = PackUtil::paddedPack(extractEven(result));
            auto hi = PackUtil::paddedPack(extractOdd(result));

            // TODO: label as rotateR64 for consistency?
            auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
            hi          = Blend::Zip32::exec(rot_hi, lo);
            lo          = Blend::Zip32::exec(lo, rot_hi);

            // Step 4 : Gather
            // Note the offset of 32 in hi
            lo = Permute::GatherWave<VW, 0>::exec(lo);
            hi = Permute::GatherWave<VW, 32>::exec(hi);

            return PackUtil::template paddedUnpack<VecSize>(concat(lo, hi));
        }
    };

    template <>
    struct AosToSoa<128, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Data comes in as AOS format:
            // There are TWO sets of VW = 8 registers (because this case BlockDim / 64 = 2):
            // 1. Vecs 0-7
            // 2. Vecs 8-15
            //
            // Register/ |          VW = 8                 |
            //     Tidx  |___0___|___1___|___...___|___7___|
            //         0 |   0   |   1   |   ...   |   7   |
            //         1 |   8   |   9   |   ...   |   15  |
            //       ... |   ... |   ... |   ...   |  ...  |
            //        63 |__504__|__505__|___...___|__511__|
            //
            // Register/ |          VW = 8                 |
            //     Tidx  |___8___|___9___|___...___|___15__|
            //         0 |  512  |  513  |   ...   |  519  |
            //         1 |  520  |  521  |   ...   |  527  |
            //       ... |   ... |   ... |   ...   |  ...  |
            //        63 |__1016_|__1017_|___...___|__1023_|

            // For each batch of VW registers
            auto v0 = extractLo(v);
            auto v1 = extractHi(v);

            // Step 1 : Unpack groups of 8
            auto r0 = unpackLoHi8(v0);
            auto r1 = unpackLoHi8(v1);

            // Step 2 : isolate data for upper 64 dim from lower 64 dim
            v0 = concat(extractLo(r0), extractLo(r1));
            v1 = concat(extractHi(r0), extractHi(r1));

            // Continue from here as if r0 and r1 are independent 64 dim.

            // Step 3 : Unpack groups of 16
            v0 = unpackLoHi16(v0);
            v1 = unpackLoHi16(v1);

            // Step 4 : Unpack groups of 32
            // In order to save some operations, we can
            // rotate the odds components only and make up the
            // offset later in gather.
            auto lo0 = PackUtil::paddedPack(extractEven(v0));
            auto hi0 = PackUtil::paddedPack(extractOdd(v0));

            auto lo1 = PackUtil::paddedPack(extractEven(v1));
            auto hi1 = PackUtil::paddedPack(extractOdd(v1));

            // TODO: label as rotateR64 for consistency?
            auto rot_hi0 = Permute::RotateWaveR<32>::exec(hi0);
            hi0          = Blend::Zip32::exec(rot_hi0, lo0);
            lo0          = Blend::Zip32::exec(lo0, rot_hi0);

            auto rot_hi1 = Permute::RotateWaveR<32>::exec(hi1);
            hi1          = Blend::Zip32::exec(rot_hi1, lo1);
            lo1          = Blend::Zip32::exec(lo1, rot_hi1);

            // Step 5 : Gather
            // Note the offset of 32 in hi
            lo0 = Permute::GatherWave<VW, 0>::exec(lo0);
            hi0 = Permute::GatherWave<VW, 32>::exec(hi0);

            lo1 = Permute::GatherWave<VW, 0>::exec(lo1);
            hi1 = Permute::GatherWave<VW, 32>::exec(hi1);

            // Step 6 : Unpack and re-order.
            auto c0 = PackUtil::template paddedUnpack<VecSize>(concat(lo0, hi0));
            //c0      = reorderEvenOdd(c0);
            c0      = concat(extractEven(c0), extractOdd(c0));
            auto c1 = PackUtil::template paddedUnpack<VecSize>(concat(lo1, hi1));
            //c1      = reorderEvenOdd(c1);
            c1 = concat(extractEven(c1), extractOdd(c1));

            return concat(c0, c1);
        }
    };

    // SOA -> AOS
    // Transform from ortho VW to inline VW
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 4> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        auto result = PackUtil::template paddedUnpack<4>(
            Permute::Scatter16<4, 0>::exec(PackUtil::paddedPack(v)));
        result = unpackLoHi4(result);
        result = unpackLoHi8(result);
        return result;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 4> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        auto hi     = (Permute::Scatter32<4, 16>::exec(PackUtil::paddedPack(extractHi(v))));
        auto lo     = (Permute::Scatter32<4, 0>::exec(PackUtil::paddedPack(extractLo(v))));
        auto result = concat(PackUtil::template paddedUnpack<2u>(lo),
                             PackUtil::template paddedUnpack<2u>(hi));
        result      = unpackLoHi8(result);
        // modified unpackLohi16
        {
            auto lo        = PackUtil::paddedPack(extractEven(result));
            auto hi        = PackUtil::paddedPack(extractOdd(result));
            auto zipped_lo = Blend::Zip16::exec(lo, hi);
            auto zipped_hi = Blend::Zip16::exec(hi, lo);
            auto rot_hi    = Swizzle::RotateR32<16>::exec(zipped_hi);
            result         = concat(PackUtil::template paddedUnpack<2u>(zipped_lo),
                            PackUtil::template paddedUnpack<2u>(rot_hi));
        }
        return result;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 8> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 4> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 2> const& v)
    {
        return 0;
    }

    template <typename DataT>
    ROCWMMA_DEVICE auto aos_soa_16x16_vw4_b32_opt(VecT<DataT, 4> const& v)
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
    ROCWMMA_DEVICE auto aos_soa_16x8_vw2_b32_opt(VecT<DataT, 2> const& v)
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
    ROCWMMA_DEVICE auto aos_soa_32x4_vw2_b32_opt(VecT<DataT, 2> const& v)
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
    ROCWMMA_DEVICE auto aos_soa_32x8_vw4_b32_opt(VecT<DataT, 4> const& v)
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
    ROCWMMA_DEVICE auto soa_aos_16x16_vw4_b32_opt(VecT<DataT, 4> const& v)
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
    ROCWMMA_DEVICE auto soa_aos_16x8_vw2_b32_opt(VecT<DataT, 2> const& v)
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

    // template<uint32_t VW>
    // struct AosToSoa<16, VW>
    // {
    //     template <typename DataT, uint32_t VecSize>
    //     ROCWMMA_DEVICE static inline decltype(auto) exec()(VecT<DataT, VecSize> const& fragData)
    //     {
    //         auto it = rocwmma::makeVectorIterator<MaxVW>(fragA.mAccess).begin();
    //         for(int i=0; i < decltype(it)::range(); i++, it++)
    //         {
    //             *it = aos_soa_16xk_b32(*it);
    //         }

    //         return fragData;

    //     };
    // };

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_IMPL_HPP
