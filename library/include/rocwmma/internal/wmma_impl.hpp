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
            __device__ static inline auto exec(RegsA&& regsA, RegsB&& regsB, RegsC& regsC)
            {
                return regsC;
            }
        };

// WMMA instructions are specific to NAVI architecture
#if ROCWMMA_ARCH_NAVI

        template <>
        struct amdgcn_wmma<float16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
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
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                    regsA.data,
                    regsB.data,
                    regsC.data,
                    0 // Result in lower 16 bits of accumulator
                    )};
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
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
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
                    KPerWmma = 16,
                };
                using ARegsT = VRegF32x8;
                using BRegsT = VRegF32x8;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                    regsA.data,
                    regsB.data,
                    regsC.data,
                    0 // Result in lower 16 bits of accumulator
                    )};
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
                    KPerWmma = 16,
                };
                using ARegsT = VRegI32x4;
                using BRegsT = VRegI32x4;
                using CRegsT = AccRegI32x8;
                using DRegsT = AccRegI32x8;
            };

            __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    1, regsA.data, 1, regsB.data, regsC.data, 1)};
                return result;
            }
        };

#endif // ROCWMMA_ARCH_NAVI

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_IMPL_HPP
