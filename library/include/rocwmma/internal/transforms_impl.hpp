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

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct AosToSoa<64, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct AosToSoa<128, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // There are TWO sets of VW = 8 registers (because this case BlockDim / 64 = 2):
            // 1. Vecs 0-7
            // 2. Vecs 8-15
            //
            // Register/ |          VW = 8                 |
            //     Tidx  |___0___|___1___|___...___|___7___|
            //         0 |   0   |   1   |   ...   |   7   |
            //         1 |   8   |   9   |   ...   |   15  |
            //       ... |   ... |   ... |   ...   |  ...  |
            //        63 |__952__|__953__|___...___|__959__|
            //
            // Register/ |          VW = 8                 |
            //     Tidx  |___8___|___9___|___...___|___15__|
            //         0 |  64   |  65   |   ...   |   71  |
            //         1 |  72   |  73   |   ...   |   79  |
            //       ... |   ... |   ... |   ...   |  ...  |
            //        63 |__1016_|__1017_|___...___|__1023_|

            // Subdivide work to each batch of WAVE_SIZE
            auto result_b0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto result_b1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(result_b0, result_b1);
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct AosToSoa<128, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct AosToSoa<256, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // There are FOUR sets of VW = 8 registers (because this case BlockDim / 64 = 4):
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
            //        63 |__1848_|__1849_|___...___|__1855_|
            //
            // Register/ |          VW = 8                 |
            //     Tidx  |___8___|___9___|___...___|___15__|
            //         0 |  64   |  65   |   ...   |   71  |
            //         1 |  72   |  73   |   ...   |   79  |
            //       ... |   ... |   ... |   ...   |  ...  |
            //        63 |__1912_|__1913_|___...___|__1919_|
            //
            // Register/ |          VW = 8                    |
            //     Tidx  |___16___|___17___|___...___|___23___|
            //         0 |  128   |  129   |   ...   |  135   |
            //         1 |  136   |  137   |   ...   |  143   |
            //       ... |   .... |   .... |   ...   |  ....  |
            //        63 |__1976__|__1977__|___...___|__1983__|
            //
            // Register/ |          VW = 8                    |
            //     Tidx  |___24___|___25___|___...___|___31___|
            //         0 |  192   |  193   |   ...   |  199   |
            //         1 |  200   |  201   |   ...   |  207   |
            //       ... |   ...  |   ...  |   ...   |  ...   |
            //        63 |__2040__|__2041__|___...___|__2047 _|

            // Extract each batch of registers and put them through the 64 size
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct AosToSoa<256, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 64;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo  = extractLo(v);
            auto hi  = extractHi(v);
            auto lo0 = extractLo(lo);
            auto lo1 = extractHi(lo);
            auto hi0 = extractLo(hi);
            auto hi1 = extractHi(hi);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
            auto v4 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
            auto v5 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
            auto v6 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
            auto v7 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

            return concat(concat(concat(v0, v1), concat(v2, v3)),
                          concat(concat(v4, v5), concat(v6, v7)));
        }
    };

#endif

    template <>
    struct AosToSoa<16, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 4;

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
        constexpr static uint32_t VecSize = 4;

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
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 4;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
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
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct AosToSoa<128, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct AosToSoa<128, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct AosToSoa<256, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct AosToSoa<256, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo  = extractLo(v);
            auto hi  = extractHi(v);
            auto lo0 = extractLo(lo);
            auto lo1 = extractHi(lo);
            auto hi0 = extractLo(hi);
            auto hi1 = extractHi(hi);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
            auto v4 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
            auto v5 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
            auto v6 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
            auto v7 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

            return concat(concat(concat(v0, v1), concat(v2, v3)),
                          concat(concat(v4, v5), concat(v6, v7)));
        }
    };

#endif

    template <>
    struct AosToSoa<16, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLoHi8
            auto unpacked_data = unpackLoHi8(v);

            // Step 2 : Gather
            unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::Gather16<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

            return unpacked_data;
        }
    };

    template <>
    struct AosToSoa<32, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLoHi16
            auto unpacked_data = unpackLoHi16(v);

            // Step 2 : Gather
            unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::Gather32<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

            return unpacked_data;
        }
    };

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<64, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 2;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

            // Step 1 : UnpackLoHi32
            auto unpacked_data = unpackLoHi32(v);

            // Step 2 : Gather
            unpacked_data = PackUtil::template paddedUnpack<2>(
                Permute::GatherWave<2, 0>::exec(PackUtil::paddedPack(unpacked_data)));

            return unpacked_data;
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<64, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");

            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{v0.data[0],
                                                    v1.data[0],
                                                    v0.data[1],
                                                    v1.data[1]};

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<128, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 4;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<128, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
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

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{unpacked_data0.data[0],
                                                    unpacked_data2.data[0],
                                                    unpacked_data0.data[1],
                                                    unpacked_data2.data[1],
                                                    unpacked_data1.data[0],
                                                    unpacked_data3.data[0],
                                                    unpacked_data1.data[1],
                                                    unpacked_data3.data[1]};

            return repack_data;
        }
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct AosToSoa<256, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = AosToSoa<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct AosToSoa<256, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
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

            // Re-pack banks
            auto repack_data = VecT<DataT, VecSize>{unpacked_data0.data[0],
                                                    unpacked_data4.data[0],
                                                    unpacked_data0.data[1],
                                                    unpacked_data4.data[1],
                                                    unpacked_data1.data[0],
                                                    unpacked_data5.data[0],
                                                    unpacked_data1.data[1],
                                                    unpacked_data5.data[1],
                                                    unpacked_data2.data[0],
                                                    unpacked_data6.data[0],
                                                    unpacked_data2.data[1],
                                                    unpacked_data6.data[1],
                                                    unpacked_data3.data[0],
                                                    unpacked_data7.data[0],
                                                    unpacked_data3.data[1],
                                                    unpacked_data7.data[1],};

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

#if ROCWMMA_WAVE64_MODE
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
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<64, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        };
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<128, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 4;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<128, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0], v.data[2]};
            auto v1 = VecT<DataT, VW>{v.data[4], v.data[6]};
            auto v2 = VecT<DataT, VW>{v.data[1], v.data[3]};
            auto v3 = VecT<DataT, VW>{v.data[5], v.data[7]};

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);
            auto unpacked_data2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v2);
            auto unpacked_data3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));
            return unpacked_data;
        };
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<256, 2>
    {
        constexpr static uint32_t VW      = 2;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<256, 2>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 2;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0], v.data[2]};
            auto v1 = VecT<DataT, VW>{v.data[4], v.data[6]};
            auto v2 = VecT<DataT, VW>{v.data[8], v.data[10]};
            auto v3 = VecT<DataT, VW>{v.data[12], v.data[14]};
            auto v4 = VecT<DataT, VW>{v.data[1], v.data[3]};
            auto v5 = VecT<DataT, VW>{v.data[5], v.data[7]};
            auto v6 = VecT<DataT, VW>{v.data[9], v.data[11]};
            auto v7 = VecT<DataT, VW>{v.data[13], v.data[15]};

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
        };
    };
#endif

    template <>
    struct SoaToAos<16, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 4;

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
        constexpr static uint32_t VecSize = 4;

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
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 4;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            using PackUtil = PackUtil<DataT>;

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
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(v));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct SoaToAos<128, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 8;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct SoaToAos<128, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo));
            auto v2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi));
            auto v3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#endif

#if ROCWMMA_WAVE64_MODE

    template <>
    struct SoaToAos<256, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };

#elif ROCWMMA_WAVE32_MODE

    template <>
    struct SoaToAos<256, 4>
    {
        constexpr static uint32_t VW      = 4;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo  = extractLo(v);
            auto hi  = extractHi(v);
            auto lo0 = extractLo(lo);
            auto lo1 = extractHi(lo);
            auto hi0 = extractLo(hi);
            auto hi1 = extractHi(hi);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo0));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo0));
            auto v2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(lo1));
            auto v3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(lo1));
            auto v4 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi0));
            auto v5 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi0));
            auto v6 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractLo(hi1));
            auto v7 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(extractHi(hi1));

            return concat(concat(concat(v0, v1), concat(v2, v3)),
                          concat(concat(v4, v5), concat(v6, v7)));
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

#if ROCWMMA_WAVE64_MODE
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
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<64, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (64 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");

            // Step 1 :  RE-PACK Banks
            auto v0 = unpackLo(extractLo(v), extractHi(v));
            auto v1 = unpackHi(extractLo(v), extractHi(v));

            // Step 2 - 4 :  Applied on VW width banks
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);

            return concat(unpacked_data0, unpacked_data1);
        };
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<128, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 16;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(v));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(v));

            return concat(v0, v1);
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<128, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (128 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            
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
            auto unpacked_data0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v0);
            auto unpacked_data1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v1);
            auto unpacked_data2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v2);
            auto unpacked_data3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_32, VW>::exec(v3);

            auto unpacked_data = concat(concat(unpacked_data0, unpacked_data1),
                                        concat(unpacked_data2, unpacked_data3));
            return unpacked_data;
        };
    };
#endif

#if ROCWMMA_WAVE64_MODE
    template <>
    struct SoaToAos<256, 8>
    {
        constexpr static uint32_t VW      = 8;
        constexpr static uint32_t VecSize = 32;

        template <typename DataT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto lo = extractLo(v);
            auto hi = extractHi(v);

            // Subdivide work to each batch of WAVE_SIZE
            auto v0 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(lo));
            auto v1 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(lo));
            auto v2 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractLo(hi));
            auto v3 = SoaToAos<Constants::AMDGCN_WAVE_SIZE_64, VW>::exec(extractHi(hi));

            return concat(concat(v0, v1), concat(v2, v3));
        }
    };
#elif ROCWMMA_WAVE32_MODE
    template <>
    struct SoaToAos<256, 8>
    {
        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            constexpr uint32_t VW = 8;
            static_assert(VecSize == VW * (256 / Constants::AMDGCN_WAVE_SIZE_32),
                          "VecSize must be specific number");
            
            // Step 1 :  RE-PACK Banks
            auto v0 = VecT<DataT, VW>{v.data[0],
                                      v.data[8],
                                      v.data[16],
                                      v.data[24],
                                      v.data[32],
                                      v.data[40],
                                      v.data[48],
                                      v.data[56]};
            auto v1 = VecT<DataT, VW>{v.data[1],
                                      v.data[9],
                                      v.data[17],
                                      v.data[25],
                                      v.data[33],
                                      v.data[41],
                                      v.data[49],
                                      v.data[57]};
            auto v2 = VecT<DataT, VW>{v.data[2],
                                      v.data[10],
                                      v.data[18],
                                      v.data[26],
                                      v.data[34],
                                      v.data[42],
                                      v.data[50],
                                      v.data[58]};                          
            auto v3 = VecT<DataT, VW>{v.data[3],
                                      v.data[11],
                                      v.data[19],
                                      v.data[27],
                                      v.data[35],
                                      v.data[43],
                                      v.data[51],
                                      v.data[59]};
            auto v4 = VecT<DataT, VW>{v.data[4],
                                      v.data[12],
                                      v.data[20],
                                      v.data[28],
                                      v.data[36],
                                      v.data[44],
                                      v.data[52],
                                      v.data[60]};
            auto v5 = VecT<DataT, VW>{v.data[5],
                                      v.data[13],
                                      v.data[21],
                                      v.data[29],
                                      v.data[37],
                                      v.data[45],
                                      v.data[53],
                                      v.data[61]};
            auto v6 = VecT<DataT, VW>{v.data[6],
                                      v.data[14],
                                      v.data[22],
                                      v.data[30],
                                      v.data[38],
                                      v.data[46],
                                      v.data[54],
                                      v.data[62]};
            auto v7 = VecT<DataT, VW>{v.data[7],
                                      v.data[15],
                                      v.data[23],
                                      v.data[31],
                                      v.data[39],
                                      v.data[47],
                                      v.data[55],
                                      v.data[63]};
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
        };
    };
#endif

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_IMPL_HPP
