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

        // TODO replace with dpp::move
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
        // TODO replace with dpp::move
        auto rot_lo = Permute::RotateWaveR<32>::exec(lo);
        auto rot_hi = Permute::RotateWaveR<32>::exec(hi);
        lo          = Blend::Zip32::exec(lo, rot_hi);
        hi          = Blend::Zip32::exec(rot_lo, hi);

        return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo),
                      PackUtil::template paddedUnpack<VecSize / 2u>(hi));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto soa_aos_wave_b32_vw4(VecT<DataT, VecSize> const& v)
    {
        using PackUtil = PackUtil<DataT>;

        // Step 1 : Scatter
        auto lo            = Permute::ScatterWave<4, 0>::exec(PackUtil::paddedPack(extractLo(v)));
        auto hi            = Permute::ScatterWave<4, 32>::exec(PackUtil::paddedPack(extractHi(v)));
        auto unpacked_data = concat(PackUtil::template paddedUnpack<2u>(lo),
                                    PackUtil::template paddedUnpack<2u>(hi));

        // Step 2 : UnpackLoHi16

        unpacked_data = PackUtil::template paddedUnpack<4>(Swizzle::RotateR32<16>::exec(
            PackUtil::paddedPack(concat(extractEven(unpacked_data), extractOdd(unpacked_data)))));

        lo          = PackUtil::paddedPack(extractLo(unpacked_data));
        hi          = PackUtil::paddedPack(extractHi(unpacked_data));
        auto rot_lo = Swizzle::RotateR32<16>::exec(lo);
        auto zip_lo = Blend::Zip16::exec(rot_lo, hi);
        auto zip_hi = Blend::Zip16::exec(hi, rot_lo);
        zip_hi      = Swizzle::RotateR32<16>::exec(zip_hi);

        unpacked_data = concat(PackUtil::template paddedUnpack<2>(zip_lo),
                               PackUtil::template paddedUnpack<2>(zip_hi));

        // Step 3 : UnpackLoHi32
        lo = PackUtil::paddedPack(extractEven(unpacked_data));
        hi = PackUtil::paddedPack(extractOdd(unpacked_data));

        zip_lo = Blend::Zip32::exec(lo, hi);
        zip_hi = Blend::Zip32::exec(hi, lo);

        auto rot_hi = Permute::RotateWaveR<32>::exec(zip_hi);

        unpacked_data = concat(PackUtil::template paddedUnpack<2u>(zip_lo),
                               PackUtil::template paddedUnpack<2u>(rot_hi));

        return unpacked_data;
    };

    template <uint32_t BlockDim, uint32_t VectorWidth>
    struct AosToSoa
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            return v;
        }
    };

    template <uint32_t BlockDim, uint32_t VectorWidth>
    struct SoaToAos
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            return v;
        }
    };

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

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<64, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil        = PackUtil<DataT>;
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");

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
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<64, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{
                v0.data[0],
                v0.data[2],
                v0.data[4],
                v0.data[6],
                v1.data[0],
                v1.data[2],
                v1.data[4],
                v1.data[6],
                v0.data[1],
                v0.data[3],
                v0.data[5],
                v0.data[7],
                v1.data[1],
                v1.data[3],
                v1.data[5],
                v1.data[7],
            };

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<128, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");
            using PackUtil = PackUtil<DataT>;

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

            // Extract each batch of registers and put them through the 64 size
            auto result_b0 = AosToSoa<64, 8>::exec(extractLo(v));
            auto result_b1 = AosToSoa<64, 8>::exec(extractHi(v));

            // Re-pack banks
            auto repacked_b0 = concat(extractEven(result_b0), extractEven(result_b1));
            auto repacked_b1 = concat(extractOdd(result_b0), extractOdd(result_b1));
            return concat(repacked_b0, repacked_b1);
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<128, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            auto lo = extractLo(v);
            auto hi = extractHi(v);
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{
                v0.data[0], v0.data[4], v1.data[0], v1.data[4], v2.data[0], v2.data[4], v3.data[0],
                v3.data[4], v0.data[1], v0.data[5], v1.data[1], v1.data[5], v2.data[1], v2.data[5],
                v3.data[1], v3.data[5], v0.data[2], v0.data[6], v1.data[2], v1.data[6], v2.data[2],
                v2.data[6], v3.data[2], v3.data[6], v0.data[3], v0.data[7], v1.data[3], v1.data[7],
                v2.data[3], v2.data[7], v3.data[3], v3.data[7],
            };

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<256, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");
            using PackUtil = PackUtil<DataT>;

            // There are FOUR sets of VW = 8 registers (because this case BlockDim / 64 = 2):
            // 1. Vecs 0-7
            // 2. Vecs 8-15
            // 3. Vecs 16-23
            // 4. Vecs 24-31
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
            //
            // Register/ |          VW = 8                    |
            //     Tidx  |___16___|___17___|___...___|___23___|
            //         0 |  1024  |  1025  |   ...   |  1031  |
            //         1 |  1032  |  1033  |   ...   |  1039  |
            //       ... |   .... |   .... |   ...   |  ....  |
            //        63 |__1528__|__1529__|___...___|__1535__|
            //
            // Register/ |          VW = 8                    |
            //     Tidx  |___24___|___25___|___...___|___31___|
            //         0 |  1536  |  1537  |   ...   |  1543  |
            //         1 |  1544  |  1545  |   ...   |  1551  |
            //       ... |   ...  |   ...  |   ...   |  ...   |
            //        63 |__2040__|__2041__|___...___|__2047 _|

            // Extract each batch of registers and put them through the 64 size
            auto lo0 = extractLo(v);
            auto hi0 = extractHi(v);

            auto result_b0 = AosToSoa<64, 8>::exec(extractLo(lo0));
            auto result_b1 = AosToSoa<64, 8>::exec(extractHi(lo0));
            auto result_b2 = AosToSoa<64, 8>::exec(extractLo(hi0));
            auto result_b3 = AosToSoa<64, 8>::exec(extractHi(hi0));

            // Re-pack banks
            return VecT<DataT, VecSize>{
                get<0>(result_b0), get<4>(result_b0), get<0>(result_b1), get<4>(result_b1),
                get<0>(result_b2), get<4>(result_b2), get<0>(result_b3), get<4>(result_b3),

                get<1>(result_b0), get<5>(result_b0), get<1>(result_b1), get<5>(result_b1),
                get<1>(result_b2), get<5>(result_b2), get<1>(result_b3), get<5>(result_b3),

                get<2>(result_b0), get<6>(result_b0), get<2>(result_b1), get<6>(result_b1),
                get<2>(result_b2), get<6>(result_b2), get<2>(result_b3), get<6>(result_b3),

                get<3>(result_b0), get<7>(result_b0), get<3>(result_b1), get<7>(result_b1),
                get<3>(result_b2), get<7>(result_b2), get<3>(result_b3), get<7>(result_b3)};
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<256, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            auto lo             = extractLo(v);
            auto hi             = extractHi(v);
            auto v0             = extractLo(lo);
            auto v1             = extractHi(lo);
            auto v2             = extractLo(hi);
            auto v3             = extractHi(hi);
            auto unpacked_data0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v0));
            auto unpacked_data1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v0));
            auto unpacked_data2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v1));
            auto unpacked_data3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v1));
            auto unpacked_data4 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v2));
            auto unpacked_data5 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v2));
            auto unpacked_data6 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v3));
            auto unpacked_data7 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v3));

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{
                unpacked_data0.data[0], unpacked_data1.data[0], unpacked_data2.data[0],
                unpacked_data3.data[0], unpacked_data4.data[0], unpacked_data5.data[0],
                unpacked_data6.data[0], unpacked_data7.data[0], unpacked_data0.data[1],
                unpacked_data1.data[1], unpacked_data2.data[1], unpacked_data3.data[1],
                unpacked_data4.data[1], unpacked_data5.data[1], unpacked_data6.data[1],
                unpacked_data7.data[1], unpacked_data0.data[2], unpacked_data1.data[2],
                unpacked_data2.data[2], unpacked_data3.data[2], unpacked_data4.data[2],
                unpacked_data5.data[2], unpacked_data6.data[2], unpacked_data7.data[2],
                unpacked_data0.data[3], unpacked_data1.data[3], unpacked_data2.data[3],
                unpacked_data3.data[3], unpacked_data4.data[3], unpacked_data5.data[3],
                unpacked_data6.data[3], unpacked_data7.data[3], unpacked_data0.data[4],
                unpacked_data1.data[4], unpacked_data2.data[4], unpacked_data3.data[4],
                unpacked_data4.data[4], unpacked_data5.data[4], unpacked_data6.data[4],
                unpacked_data7.data[4], unpacked_data0.data[5], unpacked_data1.data[5],
                unpacked_data2.data[5], unpacked_data3.data[5], unpacked_data4.data[5],
                unpacked_data5.data[5], unpacked_data6.data[5], unpacked_data7.data[5],
                unpacked_data0.data[6], unpacked_data1.data[6], unpacked_data2.data[6],
                unpacked_data3.data[6], unpacked_data4.data[6], unpacked_data5.data[6],
                unpacked_data6.data[6], unpacked_data7.data[6], unpacked_data0.data[7],
                unpacked_data1.data[7], unpacked_data2.data[7], unpacked_data3.data[7],
                unpacked_data4.data[7], unpacked_data5.data[7], unpacked_data6.data[7],
                unpacked_data7.data[7],
            };

            return repack_data;
        }
    };
#endif

    template <>
    struct AosToSoa<16, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = VW;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLoHi4
            auto unpacked_data = unpackLoHi4(v);

            // Step 2 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            // Step 3 : Gather
            unpacked_data = PackUtil::template paddedUnpack<4>(
                Permute::Gather16<4, 0>::exec(PackUtil::paddedPack(unpacked_data)));

            return unpacked_data;
        }
    };

    template <>
    struct AosToSoa<32, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = VW;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLoHi8
            auto unpacked_data = unpackLoHi8(v);

            // Step 2 : UnpackLoHi16
            auto lo       = PackUtil::paddedPack(extractEven(unpacked_data));
            auto hi       = PackUtil::paddedPack(extractOdd(unpacked_data));
            auto rot_hi   = Swizzle::RotateR32<16>::exec(hi);
            hi            = Blend::Zip16::exec(rot_hi, lo);
            lo            = Blend::Zip16::exec(lo, rot_hi);
            unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                   PackUtil::template paddedUnpack<VW / 2>(hi));

            // Step 3 : Gather
            hi = Permute::Gather32<4, 16>::exec(PackUtil::paddedPack(extractHi(unpacked_data)));
            lo = Permute::Gather32<4, 0>::exec(PackUtil::paddedPack(extractLo(unpacked_data)));
            unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                   PackUtil::template paddedUnpack<VW / 2>(hi));

            return unpacked_data;
        }
    };

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<64, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLohi16
            auto unpacked_data = unpackLoHi16(v);

            // Step 2 : UnpackLohi32
            auto lo = PackUtil::paddedPack(extractEven(unpacked_data));
            auto hi = PackUtil::paddedPack(extractOdd(unpacked_data));

            hi = Permute::RotateWaveR<32>::exec(hi);

            auto zip_lo = Blend::Zip32::exec(lo, hi);
            auto zip_hi = Blend::Zip32::exec(hi, lo);

            // Step 3 : Gather
            lo = Permute::GatherWave<4, 0>::exec(zip_lo);
            hi = Permute::GatherWave<4, 32>::exec(zip_hi);

            return concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                          PackUtil::template paddedUnpack<VW / 2>(hi));
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<64, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{v0.data[0],
                                                    v0.data[2],
                                                    v1.data[0],
                                                    v1.data[2],
                                                    v0.data[1],
                                                    v0.data[3],
                                                    v1.data[1],
                                                    v1.data[3]};

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<128, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");
            // Re-pack banks
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            auto repack_data = VecT<DataT, VecSize>{v0.data[0],
                                                    v0.data[2],
                                                    v1.data[0],
                                                    v1.data[2],
                                                    v0.data[1],
                                                    v0.data[3],
                                                    v1.data[1],
                                                    v1.data[3]};

            return repack_data;
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<128, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Re-pack banks
            auto v_lo = extractLo(v);
            auto v_hi = extractHi(v);
            auto v0   = extractLo(v_lo);
            auto v1   = extractHi(v_lo);
            auto v2   = extractLo(v_hi);
            auto v3   = extractHi(v_hi);

            // Step 1 - 3 : Applied on VW width banks
            auto unpacked_data0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);
            auto unpacked_data2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v2);
            auto unpacked_data3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v3);
            auto repack_data    = VecT<DataT, VecSize>{unpacked_data0.data[0],
                                                       unpacked_data1.data[0],
                                                       unpacked_data2.data[0],
                                                       unpacked_data3.data[0],
                                                       unpacked_data0.data[1],
                                                       unpacked_data1.data[1],
                                                       unpacked_data2.data[1],
                                                       unpacked_data3.data[1],
                                                       unpacked_data0.data[2],
                                                       unpacked_data1.data[2],
                                                       unpacked_data2.data[2],
                                                       unpacked_data3.data[2],
                                                       unpacked_data0.data[3],
                                                       unpacked_data1.data[3],
                                                       unpacked_data2.data[3],
                                                       unpacked_data3.data[3]};

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<256, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");
            // Step 4 : Re-pack banks
            auto v_lo = extractLo(v);
            auto v_hi = extractHi(v);
            auto v0   = extractLo(v_lo);
            auto v1   = extractHi(v_lo);
            auto v2   = extractLo(v_hi);
            auto v3   = extractHi(v_hi);

            // Step 1 - 3 : Applied on VW width banks
            auto unpacked_data0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v0);
            auto unpacked_data1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v1);
            auto unpacked_data2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v2);
            auto unpacked_data3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v3);

            auto repack_data = VecT<DataT, VecSize>{unpacked_data0.data[0],
                                                    unpacked_data1.data[0],
                                                    unpacked_data2.data[0],
                                                    unpacked_data3.data[0],
                                                    unpacked_data0.data[1],
                                                    unpacked_data1.data[1],
                                                    unpacked_data2.data[1],
                                                    unpacked_data3.data[1],
                                                    unpacked_data0.data[2],
                                                    unpacked_data1.data[2],
                                                    unpacked_data2.data[2],
                                                    unpacked_data3.data[2],
                                                    unpacked_data0.data[3],
                                                    unpacked_data1.data[3],
                                                    unpacked_data2.data[3],
                                                    unpacked_data3.data[3]};

            return repack_data;
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<256, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Step 4 : Re-pack banks
            auto v_lo = extractLo(v);
            auto v_hi = extractHi(v);
            auto v0   = extractLo(v_lo);
            auto v1   = extractHi(v_lo);
            auto v2   = extractLo(v_hi);
            auto v3   = extractHi(v_hi);

            // Step 1 - 3 : Applied on VW width banks
            auto unpacked_data0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v0));
            auto unpacked_data1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v0));
            auto unpacked_data2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v1));
            auto unpacked_data3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v1));
            auto unpacked_data4 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v2));
            auto unpacked_data5 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v2));
            auto unpacked_data6 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v3));
            auto unpacked_data7 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v3));

            auto repack_data = VecT<DataT, VecSize>{
                unpacked_data0.data[0], unpacked_data2.data[0], unpacked_data4.data[0],
                unpacked_data6.data[0], unpacked_data0.data[1], unpacked_data2.data[1],
                unpacked_data4.data[1], unpacked_data6.data[1], unpacked_data0.data[2],
                unpacked_data2.data[2], unpacked_data4.data[2], unpacked_data6.data[2],
                unpacked_data0.data[3], unpacked_data2.data[3], unpacked_data4.data[3],
                unpacked_data6.data[3], unpacked_data1.data[0], unpacked_data3.data[0],
                unpacked_data5.data[0], unpacked_data7.data[0], unpacked_data1.data[1],
                unpacked_data3.data[1], unpacked_data5.data[1], unpacked_data7.data[1],
                unpacked_data1.data[2], unpacked_data3.data[2], unpacked_data5.data[2],
                unpacked_data7.data[2], unpacked_data1.data[3], unpacked_data3.data[3],
                unpacked_data5.data[3], unpacked_data7.data[3]};

            return repack_data;
        }
    };
#endif

    template <>
    struct SoaToAos<16, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::Scatter16<2, 0>::exec(PackUtil::paddedPack(v)));

            // Step 2 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            return unpacked_data;
        }
    };

    template <>
    struct SoaToAos<32, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::Scatter32<2, 0>::exec(PackUtil::paddedPack(v)));

            // Step 2 : UnpackLoHi16
            unpacked_data = unpackLoHi16(unpacked_data);

            return unpacked_data;
        }
    };

    template <>
    struct SoaToAos<64, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::ScatterWave<2, 0>::exec(PackUtil::paddedPack(v)));

            // Step 2 : UnpackLoHi32
            unpacked_data = unpackLoHi32(unpacked_data);

            return unpacked_data;
        };
    };

    template <>
    struct SoaToAos<128, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 4;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<64, 2>::exec(v0);
            auto unpacked_data1 = SoaToAos<64, 2>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        }
    };

    template <>
    struct SoaToAos<256, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VecSize / 4>{v.data[0], v.data[2]};
            auto v1 = VecT<DataT, VecSize / 4>{v.data[4], v.data[6]};
            auto v2 = VecT<DataT, VecSize / 4>{v.data[1], v.data[3]};
            auto v3 = VecT<DataT, VecSize / 4>{v.data[5], v.data[7]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<64, 2>::exec(v0);
            auto unpacked_data1 = SoaToAos<64, 2>::exec(v1);
            auto unpacked_data2 = SoaToAos<64, 2>::exec(v2);
            auto unpacked_data3 = SoaToAos<64, 2>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));
            return unpacked_data;
        }
    };

    template <>
    struct SoaToAos<16, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = VW;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto unpacked_data = PackUtil::template paddedUnpack<VW>(
                Permute::Scatter16<4, 0>::exec(PackUtil::paddedPack(v)));

            // Step 2 : UnpackLoHi4
            unpacked_data = unpackLoHi4(unpacked_data);

            // Step 3 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            return unpacked_data;
        }
    };

    template <>
    struct SoaToAos<32, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = VW;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto hi = (Permute::Scatter32<4, 16>::exec(PackUtil::paddedPack(extractHi(v))));
            auto lo = (Permute::Scatter32<4, 0>::exec(PackUtil::paddedPack(extractLo(v))));
            auto unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                        PackUtil::template paddedUnpack<VW / 2>(hi));

            // Step 2 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            // Step 3 : UnpackLoHi16
            lo             = PackUtil::paddedPack(extractEven(unpacked_data));
            hi             = PackUtil::paddedPack(extractOdd(unpacked_data));
            auto zipped_lo = Blend::Zip16::exec(lo, hi);
            auto zipped_hi = Blend::Zip16::exec(hi, lo);
            auto rot_hi    = Swizzle::RotateR32<16>::exec(zipped_hi);
            unpacked_data  = concat(PackUtil::template paddedUnpack<VW / 2>(zipped_lo),
                                   PackUtil::template paddedUnpack<VW / 2>(rot_hi));

            return unpacked_data;
        }
    };

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<64, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil        = PackUtil<DataT>;
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");

            // Step 1 : Scatter
            auto lo = Permute::ScatterWave<4, 0>::exec(PackUtil::paddedPack(extractLo(v)));
            auto hi = Permute::ScatterWave<4, 32>::exec(PackUtil::paddedPack(extractHi(v)));
            auto unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(lo),
                                        PackUtil::template paddedUnpack<VW / 2>(hi));

            // Step 2 : UnpackLoHi16

            unpacked_data = PackUtil::template paddedUnpack<4>(
                Swizzle::RotateR32<16>::exec(PackUtil::paddedPack(
                    concat(extractEven(unpacked_data), extractOdd(unpacked_data)))));

            lo          = PackUtil::paddedPack(extractLo(unpacked_data));
            hi          = PackUtil::paddedPack(extractHi(unpacked_data));
            auto rot_lo = Swizzle::RotateR32<16>::exec(lo);
            auto zip_lo = Blend::Zip16::exec(rot_lo, hi);
            auto zip_hi = Blend::Zip16::exec(hi, rot_lo);
            zip_hi      = Swizzle::RotateR32<16>::exec(zip_hi);

            unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(zip_lo),
                                   PackUtil::template paddedUnpack<VW / 2>(zip_hi));

            // Step 3 : UnpackLoHi32
            lo = PackUtil::paddedPack(extractEven(unpacked_data));
            hi = PackUtil::paddedPack(extractOdd(unpacked_data));

            zip_lo = Blend::Zip32::exec(lo, hi);
            zip_hi = Blend::Zip32::exec(hi, lo);

            auto rot_hi = Permute::RotateWaveR<32>::exec(zip_hi);

            unpacked_data = concat(PackUtil::template paddedUnpack<VW / 2>(zip_lo),
                                   PackUtil::template paddedUnpack<VW / 2>(rot_hi));

            return unpacked_data;
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<64, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<128, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");

            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<128, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");

            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0], v.data[4], v.data[8], v.data[12]};
            auto v1 = VecT<DataT, VW>{v.data[1], v.data[5], v.data[9], v.data[13]};
            auto v2 = VecT<DataT, VW>{v.data[2], v.data[6], v.data[10], v.data[14]};
            auto v3 = VecT<DataT, VW>{v.data[3], v.data[7], v.data[11], v.data[15]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);
            auto unpacked_data2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v2);
            auto unpacked_data3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));
            return unpacked_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<256, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_64),
                          "VecSize must be specific number");

            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0], v.data[4], v.data[8], v.data[12]};
            auto v1 = VecT<DataT, VW>{v.data[1], v.data[5], v.data[9], v.data[13]};
            auto v2 = VecT<DataT, VW>{v.data[2], v.data[6], v.data[10], v.data[14]};
            auto v3 = VecT<DataT, VW>{v.data[3], v.data[7], v.data[11], v.data[15]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v1);
            auto unpacked_data2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v2);
            auto unpacked_data3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));
            return unpacked_data;
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<256, 4>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 4;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");

            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0], v.data[4], v.data[8], v.data[12]};
            auto v2 = VecT<DataT, VW>{v.data[1], v.data[5], v.data[9], v.data[13]};
            auto v4 = VecT<DataT, VW>{v.data[2], v.data[6], v.data[10], v.data[14]};
            auto v6 = VecT<DataT, VW>{v.data[3], v.data[7], v.data[11], v.data[15]};
            auto v1 = VecT<DataT, VW>{v.data[16], v.data[20], v.data[24], v.data[28]};
            auto v3 = VecT<DataT, VW>{v.data[17], v.data[21], v.data[25], v.data[29]};
            auto v5 = VecT<DataT, VW>{v.data[18], v.data[22], v.data[26], v.data[30]};
            auto v7 = VecT<DataT, VW>{v.data[19], v.data[23], v.data[27], v.data[31]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);
            auto unpacked_data2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v2);
            auto unpacked_data3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v3);
            auto unpacked_data4 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v4);
            auto unpacked_data5 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v5);
            auto unpacked_data6 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v6);
            auto unpacked_data7 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v7);

            auto unpacked_data = concat(concat(concat(unpacked_data0, unpacked_data1),
                                               concat(unpacked_data2, unpacked_data3)),
                                        concat(concat(unpacked_data4, unpacked_data5),
                                               concat(unpacked_data6, unpacked_data7)));
            return unpacked_data;
        }
    };
#endif

    template <>
    struct SoaToAos<16, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto unpacked_data = PackUtil::template paddedUnpack<8>(
                Permute::Scatter16<8, 0>::exec(PackUtil::paddedPack(v)));

            // Step 2 : UnpackLoHi2
            unpacked_data = unpackLoHi2(unpacked_data);

            // Step 3 : UnpackLoHi4
            unpacked_data = unpackLoHi4(unpacked_data);

            // Step 4 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            return unpacked_data;
        }
    };

    template <>
    struct SoaToAos<32, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto hi = (Permute::Scatter32<8, 16>::exec(PackUtil::paddedPack(extractHi(v))));
            auto lo = (Permute::Scatter32<8, 0>::exec(PackUtil::paddedPack(extractLo(v))));
            auto unpacked_data = concat(PackUtil::template paddedUnpack<4>(lo),
                                        PackUtil::template paddedUnpack<4>(hi));

            // Step 2 : UnpackLoHi4
            unpacked_data = unpackLoHi4(unpacked_data);

            // Step 3 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            // Step 4 : UnpackLoHi16 with half rotation
            lo = PackUtil::paddedPack(extractEven(unpacked_data));
            hi = PackUtil::paddedPack(extractOdd(unpacked_data));

            auto lo_final = Dpp::Driver<DppImpl::Ops::MaskMove, 0x5, 0xF>::exec(lo, hi);
            hi            = Dpp::Driver<DppImpl::Ops::MaskMove, 0x5, 0xF>::exec(hi, lo);

            hi = Swizzle::RotateR32<16>::exec(hi);

            return concat(PackUtil::template paddedUnpack<4u>(lo_final),
                          PackUtil::template paddedUnpack<4u>(hi));
        }
    };

    template <>
    struct SoaToAos<64, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : Scatter
            auto hi = (Permute::ScatterWave<8, 32>::exec(PackUtil::paddedPack(extractHi(v))));
            auto lo = (Permute::ScatterWave<8, 0>::exec(PackUtil::paddedPack(extractLo(v))));
            auto unpacked_data = concat(PackUtil::template paddedUnpack<4>(lo),
                                        PackUtil::template paddedUnpack<4>(hi));

            // Step 2 : UnpackLoHi8
            unpacked_data = unpackLoHi8(unpacked_data);

            // Step 3 : unpackLoHi16
            unpacked_data = unpackLoHi16(unpacked_data);

            // Step 4 : UnpackLoHi32 with half rotation
            lo = PackUtil::paddedPack(extractEven(unpacked_data));
            hi = PackUtil::paddedPack(extractOdd(unpacked_data));

            auto lo_final = Dpp::Driver<DppImpl::Ops::MaskMove, 0x3, 0xF>::exec(lo, hi);
            hi            = Dpp::Driver<DppImpl::Ops::MaskMove, 0x3, 0xF>::exec(hi, lo);

            hi = Permute::RotateWaveR<32>::exec(hi);

            return concat(PackUtil::template paddedUnpack<VecSize / 2u>(lo_final),
                          PackUtil::template paddedUnpack<VecSize / 2u>(hi));
        };
    };

    template <>
    struct SoaToAos<128, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<64, 8>::exec(v0);
            auto unpacked_data1 = SoaToAos<64, 8>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        }
    };

    template <>
    struct SoaToAos<256, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0],
                                      v.data[8],
                                      v.data[16],
                                      v.data[24],
                                      v.data[1],
                                      v.data[9],
                                      v.data[17],
                                      v.data[25]};
            auto v1 = VecT<DataT, VW>{v.data[2],
                                      v.data[10],
                                      v.data[18],
                                      v.data[26],
                                      v.data[3],
                                      v.data[11],
                                      v.data[19],
                                      v.data[27]};
            auto v2 = VecT<DataT, VW>{v.data[4],
                                      v.data[12],
                                      v.data[20],
                                      v.data[28],
                                      v.data[5],
                                      v.data[13],
                                      v.data[21],
                                      v.data[29]};
            auto v3 = VecT<DataT, VW>{v.data[6],
                                      v.data[14],
                                      v.data[22],
                                      v.data[30],
                                      v.data[7],
                                      v.data[15],
                                      v.data[23],
                                      v.data[31]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<64, 8>::exec(v0);
            auto unpacked_data1 = SoaToAos<64, 8>::exec(v1);
            auto unpacked_data2 = SoaToAos<64, 8>::exec(v2);
            auto unpacked_data3 = SoaToAos<64, 8>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));

            return unpacked_data;
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
        return 0;
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
        return 0;
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
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 4> const& v0,
                                                        VecT<DataT, 4> const& v1)
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
