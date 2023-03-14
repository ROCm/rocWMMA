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
#ifndef ROCWMMA_WMMA_IMPL_HPP
#define ROCWMMA_WMMA_IMPL_HPP

#include "permute.hpp"

namespace rocwmma
{

    namespace detail
    {
        template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN>
        struct amdgcn_wmma
        {
            template <typename RegsA, typename RegsB, typename RegsC>
            ROCWMMA_DEVICE static inline auto exec(RegsA&& regsA, RegsB&& regsB, RegsC& regsC)
            {
                return regsC;
            }
        };

        template <typename ComputeT, uint32_t AccumBits, typename Enabler = void>
        struct AccumAdapter
        {
            template <typename IncomingT>
            ROCWMMA_DEVICE static inline auto unpack(IncomingT&& accumVec)
            {
                // No unpack needed
                return accumVec;
            }

            template <typename IncomingT>
            ROCWMMA_DEVICE static inline auto pack(IncomingT&& accumVec)
            {
                // No pack needed
                return accumVec;
            }
        };

// WMMA instructions are specific to gfx11 architecture
#if ROCWMMA_ARCH_NAVI

        struct WmmaCtrlFlags
        {
            enum : uint32_t
            {
                // Output register selection of WMMA.
                // Low = bits [15:0]
                // High = bits[31:16]
                LOW  = 0,
                HIGH = 1,

                // Signage indicator of inputs / accum
                UNSIGNED = 0,
                SIGNED   = 1
            };
        };

        template <>
        struct amdgcn_wmma<float16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma  = 16,
                    InputSign = WmmaCtrlFlags::SIGNED,
                    AccumBits = WmmaCtrlFlags::LOW,
                    AccumSign = WmmaCtrlFlags::SIGNED
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<float16_t, float16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma  = 16,
                    InputSign = WmmaCtrlFlags::SIGNED,
                    AccumBits = WmmaCtrlFlags::LOW,
                    AccumSign = WmmaCtrlFlags::SIGNED
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    regsA.data, regsB.data, regsC.data, Traits::AccumBits)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<hfloat16_t, float32_t, 16, 16>
            : public amdgcn_wmma<float16_t, float32_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_wmma<hfloat16_t, hfloat16_t, 16, 16>
            : public amdgcn_wmma<float16_t, float16_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_wmma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma  = 16,
                    InputSign = WmmaCtrlFlags::SIGNED,
                    AccumBits = WmmaCtrlFlags::LOW,
                    AccumSign = WmmaCtrlFlags::SIGNED
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma  = 16,
                    InputSign = WmmaCtrlFlags::SIGNED,
                    AccumBits = WmmaCtrlFlags::LOW,
                    AccumSign = WmmaCtrlFlags::SIGNED
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                    regsA.data, regsB.data, regsC.data, Traits::AccumBits)};
                return result;
            }
        };

        template <>
        struct amdgcn_wmma<int8_t, int32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma  = 16,
                    InputSign = WmmaCtrlFlags::SIGNED,
                    AccumBits = WmmaCtrlFlags::LOW,
                    AccumSign = WmmaCtrlFlags::SIGNED
                };
                using ARegsT = VRegI32x4;
                using BRegsT = VRegI32x4;
                using CRegsT = AccRegI32x8;
                using DRegsT = AccRegI32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(Traits::InputSign,
                                                                          regsA.data,
                                                                          Traits::InputSign,
                                                                          regsB.data,
                                                                          regsC.data,
                                                                          Traits::AccumSign)};
                return result;
            }
        };

        // Accumulator data needs some special treatment for data types < 4 Byte, due to unpacked layout AND
        // variable element placement within 32b element containers.
        // This adapter supplies the correct accumulator layout for small data types.
        template <typename ComputeT, uint32_t AccumBits>
        struct AccumAdapter<ComputeT,
                            AccumBits,
                            typename std::enable_if_t<(PackTraits<ComputeT>::PackRatio > 1)>>
        {
            using PackTraits = PackTraits<ComputeT>;
            using UnpackedT  = typename PackTraits::UnpackedT;
            using PackedT    = typename PackTraits::PackedT;

            template <typename IncomingT>
            ROCWMMA_DEVICE static inline auto unpack(IncomingT const& accumVec)
            {
                // Accum data is comes in packed. WMMA accumulator needs unpacked data.
                using VecTraitsIn = VecTraits<IncomingT>;
                using Unpacker    = Unpack<ComputeT>;

                static_assert(std::is_same<PackedT, typename VecTraitsIn::DataT>::value,
                              "Unexpected incoming packed type");

                // WMMA accumulator operates on unpacked data in separate 32b elements.
                // In the case of f16, what needs to happen is extend each unpacked element to 32b wide
                // and shift the 16b data to the correct spot (determined by the WMMA backend).
                // The nasty bit is that due of the extended 32b element size, the final accumulation vector
                // is masqueraded as a 'packed' type, but with the same vector size as unpacked.
                using AccumExtVecT = typename VecTraitsIn::
                    template VecT<PackedT, VecTraitsIn::size() * PackTraits::PackRatio>;

                // First step, unpack. This is not yet 32b wide.
                auto unpacked = Unpacker::exec(accumVec);

                // Next, create the destination 32b wide elements
                auto accum = AccumExtVecT();

                // Iterate over both vectors, one read, one write
                auto const rIt = makeVectorIterator(unpacked).begin();
                auto       wIt = makeVectorIterator(accum).begin();

                static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                              "Unexpected iterator range mismatch");

                // Don't convert, however emplace element data in upper / lower halves of accum.
#pragma unroll
                for(uint32_t i = 0u; i < decltype(rIt)::range(); i++)
                {
                    union
                    {
                        PackedT   packed;
                        UnpackedT unpacked[PackTraits::PackRatio];
                    } a = {PackedT(0)};

                    a.unpacked[AccumBits] = get<0>(*rIt);
                    get<0>(*wIt)          = a.packed;

                    wIt++;
                    rIt++;
                }

                return accum;
            }

            template <typename IncomingT>
            ROCWMMA_DEVICE static inline auto pack(IncomingT const& accumVec)
            {
                // Accum data comes out as unpacked, but with extended 32b elements
                using VecTraitsIn = VecTraits<IncomingT>;
                using Packer      = Pack<ComputeT>;

                static_assert(std::is_same<PackedT, typename VecTraitsIn::DataT>::value,
                              "Unexpected incoming unpacked type");

                // As before, WMMA accumulator operates on unpacked data in separate 32b elements.
                // In the case of f16, what needs to happen is shrink each unpacked 32b element the original
                // 16b size, preserving the 16b data from correct spot (determined by the WMMA backend).
                auto accum = typename Packer::Traits::InputT();

                auto const rIt = makeVectorIterator(accumVec).begin();
                auto       wIt = makeVectorIterator(accum).begin();

                static_assert(decltype(rIt)::range() == decltype(wIt)::range(),
                              "Unexpected iterator range mismatch");

                // Don't convert, however pick element data in upper / lower halves of accumVec.
#pragma unroll
                for(uint32_t i = 0u; i < decltype(rIt)::range(); i++)
                {
                    union
                    {
                        PackedT   packed;
                        UnpackedT unpacked[PackTraits::PackRatio];
                    } a = {PackedT(0)};

                    a.packed     = get<0>(*rIt);
                    get<0>(*wIt) = a.unpacked[AccumBits];

                    wIt++;
                    rIt++;
                }

                return Packer::exec(accum);
            }
        };

#endif // ROCWMMA_ARCH_NAVI

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_IMPL_HPP
