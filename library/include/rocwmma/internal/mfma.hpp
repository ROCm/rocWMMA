/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_MFMA_HPP
#define ROCWMMA_MFMA_HPP

#include "convert.hpp"
#include "io_traits.hpp"
#include "pack.hpp"
#include "types.hpp"
#include "unpack.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN>
        struct amdgcn_mfma;

        template <>
        struct amdgcn_mfma<float16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_16x16x16f16(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<float16_t, float32_t, 16, 16>;
                using UnpackC           = Unpack<float16_t, 2>;
                using PackD             = Pack<float16_t, 4>;
                using Convert_fp16_fp32 = Convert<float16_t, float32_t>;
                using Convert_fp32_fp16 = Convert<float32_t, float16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D
                // to fp16 as 'simulated' fp16 computation
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_fp16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_32x32x8f16(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<float16_t, float32_t, 32, 32>;
                using UnpackC           = Unpack<float16_t, 8>;
                using PackD             = Pack<float16_t, 16>;
                using Convert_fp16_fp32 = Convert<float16_t, float32_t>;
                using Convert_fp32_fp16 = Convert<float32_t, float16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to fp16 result;
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_fp16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, float32_t, 16, 16>
            : public amdgcn_mfma<float16_t, float32_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, 16, 16>
            : public amdgcn_mfma<float16_t, float16_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, float32_t, 32, 32>
            : public amdgcn_mfma<float16_t, float32_t, 32, 32>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, 32, 32>
            : public amdgcn_mfma<float16_t, float16_t, 32, 32>
        {
        };

#if !__gfx908__

        // NOTE: Successors to gfx908 have upgraded bf16 instructions
        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 16, 16>;
                using UnpackC           = Unpack<bfloat16_t, 2>;
                using PackD             = Pack<bfloat16_t, 4>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_bf16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 32, 32>;
                using UnpackC           = Unpack<bfloat16_t, 8>;
                using PackD             = Pack<bfloat16_t, 16>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_bf16::exec(Dfp32));
            }
        };

#else // !__gfx908___

        // NOTE: gfx908 architecture supports only subset of bf16 instructions
        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decltype(regsA)),
                              "Inconsistent data formats");
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_16x16x8bf16(*reinterpret_cast<TypeIn const&>(regsA),
                                                          *reinterpret_cast<TypeIn const&>(regsB),
                                                          *regsC,
                                                          0,
                                                          0,
                                                          0));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 16, 16>;
                using UnpackC           = Unpack<bfloat16_t, 2>;
                using PackD             = Pack<bfloat16_t, 4>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_bf16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decltype(regsA)),
                              "Inconsistent data formats");
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_32x32x4bf16(*reinterpret_cast<TypeIn const&>(regsA),
                                                          *reinterpret_cast<TypeIn const&>(regsB),
                                                          *regsC,
                                                          0,
                                                          0,
                                                          0));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 32, 32>;
                using UnpackC           = Unpack<bfloat16_t, 8>;
                using PackD             = Pack<bfloat16_t, 16>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(UnpackC::exec(regsC)));
                return PackD::exec(Convert_fp32_bf16::exec(Dfp32));
            }
        };

#endif // !__gfx908___

        template <>
        struct amdgcn_mfma<float32_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_16x16x4f32(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        // Single 32 x 32 block mfma
        template <>
        struct amdgcn_mfma<float32_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 2
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f32_32x32x2f32(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegI32x1;
                using BRegsT = VRegI32x1;
                using CRegsT = AccRegI32x4;
                using DRegsT = AccRegI32x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_i32_16x16x16i8(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegI32x1;
                using BRegsT = VRegI32x1;
                using CRegsT = AccRegI32x16;
                using DRegsT = AccRegI32x16;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_i32_32x32x8i8(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

#if !__gfx908__

        // NOTE: Successors to gfx908 support fp64 mfma
        template <>
        struct amdgcn_mfma<float64_t, float64_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF64x1;
                using BRegsT = VRegF64x1;
                using CRegsT = AccRegF64x4;
                using DRegsT = AccRegF64x4;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return typename Traits::DRegsT(
                    __builtin_amdgcn_mfma_f64_16x16x4f64(*regsA, *regsB, *regsC, 0, 0, 0));
            }
        };

#else

        // Required for general fp64 support
        template <>
        struct amdgcn_mfma<float64_t, float64_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF64x1;
                using BRegsT = VRegF64x1;
                using CRegsT = AccRegF64x4;
                using DRegsT = AccRegF64x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // MI-100 lacks support for fp64 MFMA instructions.
            __attribute__((deprecated("fp64 mfma not supported on MI-100")))
            __device__ static inline auto
                exec(typename Traits::ARegsT const& regsA,
                     typename Traits::BRegsT const& regsB,
                     typename Traits::CRegsT const& regsC)

                    -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

#endif // !__gfx908__

    } // namespace detail

    template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct Mfma
    {
        using IOTraitsA   = IOTraits<BlockM, BlockK, InputT>;
        using IOTraitsB   = IOTraits<BlockK, BlockN, InputT>;
        using IOTraitsAcc = IOTraits<BlockM, BlockN, ComputeT>;
        struct Traits
        {
            using MFMA = detail::amdgcn_mfma<InputT, ComputeT, BlockM, BlockN>;

            enum : uint32_t
            {
                MfmaCount = BlockK / MFMA::Traits::KPerMfma,
                MinK      = MFMA::Traits::KPerMfma,
            };

            // Propagate individual MFMA types to full block inputs.
            using ARegsT = VecT<typename MFMA::Traits::ARegsT::DataT,
                                MfmaCount * MFMA::Traits::ARegsT::size()>;
            using BRegsT = VecT<typename MFMA::Traits::BRegsT::DataT,
                                MfmaCount * MFMA::Traits::BRegsT::size()>;
            using CRegsT = VecT<typename MFMA::Traits::CRegsT::DataT, MFMA::Traits::CRegsT::size()>;
            using DRegsT = VecT<typename MFMA::Traits::DRegsT::DataT, MFMA::Traits::DRegsT::size()>;

            // Sanity checks
            static_assert(BlockK >= MinK, "BlockK is not a minimum of MinK");
            static_assert(BlockK % MinK == 0, "BlockK is not a multiple of MinK");
            static_assert(std::is_same<ARegsT, BRegsT>::value,
                          "A and B registers must be of same type");
            static_assert(std::is_same<CRegsT, DRegsT>::value,
                          "C and D registers must be of same type");
            static_assert(ARegsT::size() == IOTraitsA::PackedSize,
                          "Unexpected packed vector size for A");
            static_assert(BRegsT::size() == IOTraitsB::PackedSize,
                          "Unexpected packed vector size for B");
            static_assert(CRegsT::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for C");
            static_assert(DRegsT::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for D");
        };

        __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                           typename Traits::BRegsT const& regsB,
                                           typename Traits::CRegsT const& regsC) ->
            typename Traits::DRegsT
        {
            typename Traits::DRegsT result = regsC;

            // Accumulate into result regs
            auto aIt = regsA.template begin<Traits::MFMA::Traits::ARegsT::size()>();
            auto bIt = regsB.template begin<Traits::MFMA::Traits::BRegsT::size()>();
#pragma unroll
            for(unsigned i = 0; i < Traits::MfmaCount; i++)
            {
                result = Traits::MFMA::exec(*aIt, *bIt, result);
                aIt++;
                bIt++;
            }
            return result;
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_MFMA_HPP
