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

#ifndef ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
#define ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP

#include "hip_device.hpp"
#include "transforms.hpp"
#include <rocwmma/rocwmma.hpp>

static constexpr uint32_t ERROR_VALUE   = 7u;
static constexpr uint32_t SUCCESS_VALUE = 0u;

namespace rocwmma
{
    namespace detail
    {
        ROCWMMA_DEVICE static inline auto threadId()
        {
            return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x)
                   + threadIdx.x;
        }
    }
    template <typename DataT, uint32_t VW, uint32_t K>
    struct AosVec;

    template <typename DataT, uint32_t VW, uint32_t K>
    struct SoaVec;

    template <typename DataT>
    struct AosVec<DataT, 8, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 8;
            constexpr uint32_t K      = 16;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) * VW + waveOffset;
            VecType        v          = {
                start, start + 1, start + 2, start + 3, start + 4, start + 5, start + 6, start + 7};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 8;
            constexpr uint32_t K      = 32;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) * VW + waveOffset;
            VecType        v          = {
                start, start + 1, start + 2, start + 3, start + 4, start + 5, start + 6, start + 7};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 64;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start,
                                             start + 1,
                                             start + 2,
                                             start + 3,
                                             start + 4,
                                             start + 5,
                                             start + 6,
                                             start + 7};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + 4 + VW * WAVE_SIZE,
                    start + 5 + VW * WAVE_SIZE,
                    start + 6 + VW * WAVE_SIZE,
                    start + 7 + VW * WAVE_SIZE,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 128;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + 4 + VW * WAVE_SIZE,
                    start + 5 + VW * WAVE_SIZE,
                    start + 6 + VW * WAVE_SIZE,
                    start + 7 + VW * WAVE_SIZE,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + 4 + VW * WAVE_SIZE,
                    start + 5 + VW * WAVE_SIZE,
                    start + 6 + VW * WAVE_SIZE,
                    start + 7 + VW * WAVE_SIZE,
                    start + VW * WAVE_SIZE * 2,
                    start + 1 + VW * WAVE_SIZE * 2,
                    start + 2 + VW * WAVE_SIZE * 2,
                    start + 3 + VW * WAVE_SIZE * 2,
                    start + 4 + VW * WAVE_SIZE * 2,
                    start + 5 + VW * WAVE_SIZE * 2,
                    start + 6 + VW * WAVE_SIZE * 2,
                    start + 7 + VW * WAVE_SIZE * 2,
                    start + VW * WAVE_SIZE * 3,
                    start + 1 + VW * WAVE_SIZE * 3,
                    start + 2 + VW * WAVE_SIZE * 3,
                    start + 3 + VW * WAVE_SIZE * 3,
                    start + 4 + VW * WAVE_SIZE * 3,
                    start + 5 + VW * WAVE_SIZE * 3,
                    start + 6 + VW * WAVE_SIZE * 3,
                    start + 7 + VW * WAVE_SIZE * 3,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 256>
    {
        ROCWMMA_DEVICE static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 256;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + 4 + VW * WAVE_SIZE,
                    start + 5 + VW * WAVE_SIZE,
                    start + 6 + VW * WAVE_SIZE,
                    start + 7 + VW * WAVE_SIZE,
                    start + VW * WAVE_SIZE * 2,
                    start + 1 + VW * WAVE_SIZE * 2,
                    start + 2 + VW * WAVE_SIZE * 2,
                    start + 3 + VW * WAVE_SIZE * 2,
                    start + 4 + VW * WAVE_SIZE * 2,
                    start + 5 + VW * WAVE_SIZE * 2,
                    start + 6 + VW * WAVE_SIZE * 2,
                    start + 7 + VW * WAVE_SIZE * 2,
                    start + VW * WAVE_SIZE * 3,
                    start + 1 + VW * WAVE_SIZE * 3,
                    start + 2 + VW * WAVE_SIZE * 3,
                    start + 3 + VW * WAVE_SIZE * 3,
                    start + 4 + VW * WAVE_SIZE * 3,
                    start + 5 + VW * WAVE_SIZE * 3,
                    start + 6 + VW * WAVE_SIZE * 3,
                    start + 7 + VW * WAVE_SIZE * 3,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + 4 + VW * WAVE_SIZE,
                    start + 5 + VW * WAVE_SIZE,
                    start + 6 + VW * WAVE_SIZE,
                    start + 7 + VW * WAVE_SIZE,
                    start + VW * WAVE_SIZE * 2,
                    start + 1 + VW * WAVE_SIZE * 2,
                    start + 2 + VW * WAVE_SIZE * 2,
                    start + 3 + VW * WAVE_SIZE * 2,
                    start + 4 + VW * WAVE_SIZE * 2,
                    start + 5 + VW * WAVE_SIZE * 2,
                    start + 6 + VW * WAVE_SIZE * 2,
                    start + 7 + VW * WAVE_SIZE * 2,
                    start + VW * WAVE_SIZE * 3,
                    start + 1 + VW * WAVE_SIZE * 3,
                    start + 2 + VW * WAVE_SIZE * 3,
                    start + 3 + VW * WAVE_SIZE * 3,
                    start + 4 + VW * WAVE_SIZE * 3,
                    start + 5 + VW * WAVE_SIZE * 3,
                    start + 6 + VW * WAVE_SIZE * 3,
                    start + 7 + VW * WAVE_SIZE * 3,
                    start + VW * WAVE_SIZE * 4,
                    start + 1 + VW * WAVE_SIZE * 4,
                    start + 2 + VW * WAVE_SIZE * 4,
                    start + 3 + VW * WAVE_SIZE * 4,
                    start + 4 + VW * WAVE_SIZE * 4,
                    start + 5 + VW * WAVE_SIZE * 4,
                    start + 6 + VW * WAVE_SIZE * 4,
                    start + 7 + VW * WAVE_SIZE * 4,
                    start + VW * WAVE_SIZE * 5,
                    start + 1 + VW * WAVE_SIZE * 5,
                    start + 2 + VW * WAVE_SIZE * 5,
                    start + 3 + VW * WAVE_SIZE * 5,
                    start + 4 + VW * WAVE_SIZE * 5,
                    start + 5 + VW * WAVE_SIZE * 5,
                    start + 6 + VW * WAVE_SIZE * 5,
                    start + 7 + VW * WAVE_SIZE * 5,
                    start + VW * WAVE_SIZE * 6,
                    start + 1 + VW * WAVE_SIZE * 6,
                    start + 2 + VW * WAVE_SIZE * 6,
                    start + 3 + VW * WAVE_SIZE * 6,
                    start + 4 + VW * WAVE_SIZE * 6,
                    start + 5 + VW * WAVE_SIZE * 6,
                    start + 6 + VW * WAVE_SIZE * 6,
                    start + 7 + VW * WAVE_SIZE * 6,
                    start + VW * WAVE_SIZE * 7,
                    start + 1 + VW * WAVE_SIZE * 7,
                    start + 2 + VW * WAVE_SIZE * 7,
                    start + 3 + VW * WAVE_SIZE * 7,
                    start + 4 + VW * WAVE_SIZE * 7,
                    start + 5 + VW * WAVE_SIZE * 7,
                    start + 6 + VW * WAVE_SIZE * 7,
                    start + 7 + VW * WAVE_SIZE * 7,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 4;
            constexpr uint32_t K      = 16;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) * VW + waveOffset;
            VecType        v          = {start, start + 1, start + 2, start + 3};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 4;
            constexpr uint32_t K      = 32;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) * VW + waveOffset;
            VecType        v          = {start, start + 1, start + 2, start + 3};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 64;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start, start + 1, start + 2, start + 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start,
                                             start + 1,
                                             start + 2,
                                             start + 3,
                                             start + VW * WAVE_SIZE,
                                             start + 1 + VW * WAVE_SIZE,
                                             start + 2 + VW * WAVE_SIZE,
                                             start + 3 + VW * WAVE_SIZE};
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 128;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start,
                                             start + 1,
                                             start + 2,
                                             start + 3,
                                             start + VW * WAVE_SIZE,
                                             start + 1 + VW * WAVE_SIZE,
                                             start + 2 + VW * WAVE_SIZE,
                                             start + 3 + VW * WAVE_SIZE};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start,
                                             start + 1,
                                             start + 2,
                                             start + 3,
                                             start + VW * WAVE_SIZE,
                                             start + 1 + VW * WAVE_SIZE,
                                             start + 2 + VW * WAVE_SIZE,
                                             start + 3 + VW * WAVE_SIZE,
                                             start + VW * WAVE_SIZE * 2,
                                             start + 1 + VW * WAVE_SIZE * 2,
                                             start + 2 + VW * WAVE_SIZE * 2,
                                             start + 3 + VW * WAVE_SIZE * 2,
                                             start + VW * WAVE_SIZE * 3,
                                             start + 1 + VW * WAVE_SIZE * 3,
                                             start + 2 + VW * WAVE_SIZE * 3,
                                             start + 3 + VW * WAVE_SIZE * 3};
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 256>
    {
        ROCWMMA_DEVICE static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 256;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {start,
                                             start + 1,
                                             start + 2,
                                             start + 3,
                                             start + VW * WAVE_SIZE,
                                             start + 1 + VW * WAVE_SIZE,
                                             start + 2 + VW * WAVE_SIZE,
                                             start + 3 + VW * WAVE_SIZE,
                                             start + VW * WAVE_SIZE * 2,
                                             start + 1 + VW * WAVE_SIZE * 2,
                                             start + 2 + VW * WAVE_SIZE * 2,
                                             start + 3 + VW * WAVE_SIZE * 2,
                                             start + VW * WAVE_SIZE * 3,
                                             start + 1 + VW * WAVE_SIZE * 3,
                                             start + 2 + VW * WAVE_SIZE * 3,
                                             start + 3 + VW * WAVE_SIZE * 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) * VW + waveOffset;
                VecType        v          = {
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + VW * WAVE_SIZE,
                    start + 1 + VW * WAVE_SIZE,
                    start + 2 + VW * WAVE_SIZE,
                    start + 3 + VW * WAVE_SIZE,
                    start + VW * WAVE_SIZE * 2,
                    start + 1 + VW * WAVE_SIZE * 2,
                    start + 2 + VW * WAVE_SIZE * 2,
                    start + 3 + VW * WAVE_SIZE * 2,
                    start + VW * WAVE_SIZE * 3,
                    start + 1 + VW * WAVE_SIZE * 3,
                    start + 2 + VW * WAVE_SIZE * 3,
                    start + 3 + VW * WAVE_SIZE * 3,
                    start + VW * WAVE_SIZE * 4,
                    start + 1 + VW * WAVE_SIZE * 4,
                    start + 2 + VW * WAVE_SIZE * 4,
                    start + 3 + VW * WAVE_SIZE * 4,
                    start + VW * WAVE_SIZE * 5,
                    start + 1 + VW * WAVE_SIZE * 5,
                    start + 2 + VW * WAVE_SIZE * 5,
                    start + 3 + VW * WAVE_SIZE * 5,
                    start + VW * WAVE_SIZE * 6,
                    start + 1 + VW * WAVE_SIZE * 6,
                    start + 2 + VW * WAVE_SIZE * 6,
                    start + 3 + VW * WAVE_SIZE * 6,
                    start + VW * WAVE_SIZE * 7,
                    start + 1 + VW * WAVE_SIZE * 7,
                    start + 2 + VW * WAVE_SIZE * 7,
                    start + 3 + VW * WAVE_SIZE * 7,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    // SoaVec
    template <typename DataT>
    struct SoaVec<DataT, 8, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 8;
            constexpr uint32_t K      = 16;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) % K + waveOffset;
            VecType        v          = {
                start,
                K + start,
                K * 2 + start,
                K * 3 + start,
                K * 4 + start,
                K * 5 + start,
                K * 6 + start,
                K * 7 + start,
            };
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 8;
            constexpr uint32_t K      = 32;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) % K + waveOffset;
            VecType        v          = {start,
                                         K + start,
                                         K * 2 + start,
                                         K * 3 + start,
                                         K * 4 + start,
                                         K * 5 + start,
                                         K * 6 + start,
                                         K * 7 + start};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 64;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE + K * 4 + start,
                    WAVE_SIZE + K * 5 + start,
                    WAVE_SIZE + K * 6 + start,
                    WAVE_SIZE + K * 7 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 128;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE + K * 4 + start,
                    WAVE_SIZE + K * 5 + start,
                    WAVE_SIZE + K * 6 + start,
                    WAVE_SIZE + K * 7 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE + K * 4 + start,
                    WAVE_SIZE + K * 5 + start,
                    WAVE_SIZE + K * 6 + start,
                    WAVE_SIZE + K * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 2 + K * 4 + start,
                    WAVE_SIZE * 2 + K * 5 + start,
                    WAVE_SIZE * 2 + K * 6 + start,
                    WAVE_SIZE * 2 + K * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                    WAVE_SIZE * 3 + K * 4 + start,
                    WAVE_SIZE * 3 + K * 5 + start,
                    WAVE_SIZE * 3 + K * 6 + start,
                    WAVE_SIZE * 3 + K * 7 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 256>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            constexpr uint32_t K        = 256;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE + K * 4 + start,
                    WAVE_SIZE + K * 5 + start,
                    WAVE_SIZE + K * 6 + start,
                    WAVE_SIZE + K * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 2 + K * 4 + start,
                    WAVE_SIZE * 2 + K * 5 + start,
                    WAVE_SIZE * 2 + K * 6 + start,
                    WAVE_SIZE * 2 + K * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                    WAVE_SIZE * 3 + K * 4 + start,
                    WAVE_SIZE * 3 + K * 5 + start,
                    WAVE_SIZE * 3 + K * 6 + start,
                    WAVE_SIZE * 3 + K * 7 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    K * 4 + start,
                    K * 5 + start,
                    K * 6 + start,
                    K * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE + K * 4 + start,
                    WAVE_SIZE + K * 5 + start,
                    WAVE_SIZE + K * 6 + start,
                    WAVE_SIZE + K * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 2 + K * 4 + start,
                    WAVE_SIZE * 2 + K * 5 + start,
                    WAVE_SIZE * 2 + K * 6 + start,
                    WAVE_SIZE * 2 + K * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                    WAVE_SIZE * 3 + K * 4 + start,
                    WAVE_SIZE * 3 + K * 5 + start,
                    WAVE_SIZE * 3 + K * 6 + start,
                    WAVE_SIZE * 3 + K * 7 + start,
                    WAVE_SIZE * 4 + start,
                    WAVE_SIZE * 4 + K + start,
                    WAVE_SIZE * 4 + K * 2 + start,
                    WAVE_SIZE * 4 + K * 3 + start,
                    WAVE_SIZE * 4 + K * 4 + start,
                    WAVE_SIZE * 4 + K * 5 + start,
                    WAVE_SIZE * 4 + K * 6 + start,
                    WAVE_SIZE * 4 + K * 7 + start,
                    WAVE_SIZE * 5 + start,
                    WAVE_SIZE * 5 + K + start,
                    WAVE_SIZE * 5 + K * 2 + start,
                    WAVE_SIZE * 5 + K * 3 + start,
                    WAVE_SIZE * 5 + K * 4 + start,
                    WAVE_SIZE * 5 + K * 5 + start,
                    WAVE_SIZE * 5 + K * 6 + start,
                    WAVE_SIZE * 5 + K * 7 + start,
                    WAVE_SIZE * 6 + start,
                    WAVE_SIZE * 6 + K + start,
                    WAVE_SIZE * 6 + K * 2 + start,
                    WAVE_SIZE * 6 + K * 3 + start,
                    WAVE_SIZE * 6 + K * 4 + start,
                    WAVE_SIZE * 6 + K * 5 + start,
                    WAVE_SIZE * 6 + K * 6 + start,
                    WAVE_SIZE * 6 + K * 7 + start,
                    WAVE_SIZE * 7 + start,
                    WAVE_SIZE * 7 + K + start,
                    WAVE_SIZE * 7 + K * 2 + start,
                    WAVE_SIZE * 7 + K * 3 + start,
                    WAVE_SIZE * 7 + K * 4 + start,
                    WAVE_SIZE * 7 + K * 5 + start,
                    WAVE_SIZE * 7 + K * 6 + start,
                    WAVE_SIZE * 7 + K * 7 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 4;
            constexpr uint32_t K      = 16;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) % K + waveOffset;
            VecType        v          = {start, K + start, K * 2 + start, K * 3 + start};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW     = 4;
            constexpr uint32_t K      = 32;
            using VecType             = VecT<DataT, VW>;
            auto           threadId   = (uint8_t)detail::threadId();
            const uint32_t waveOffset = threadId / K * VW * K;
            auto           start      = (threadId % K) % K + waveOffset;
            VecType        v          = {start, K + start, K * 2 + start, K * 3 + start};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 64;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 128;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 256>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            constexpr uint32_t K        = 256;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType             = VecT<DataT, VecSize>;
                const uint32_t waveOffset = threadId / WAVE_SIZE * VW * K;
                auto           start      = (threadId % WAVE_SIZE) % K + waveOffset;
                VecType        v          = {
                    start,
                    K + start,
                    K * 2 + start,
                    K * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + K + start,
                    WAVE_SIZE + K * 2 + start,
                    WAVE_SIZE + K * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + K + start,
                    WAVE_SIZE * 2 + K * 2 + start,
                    WAVE_SIZE * 2 + K * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + K + start,
                    WAVE_SIZE * 3 + K * 2 + start,
                    WAVE_SIZE * 3 + K * 3 + start,
                    WAVE_SIZE * 4 + start,
                    WAVE_SIZE * 4 + K + start,
                    WAVE_SIZE * 4 + K * 2 + start,
                    WAVE_SIZE * 4 + K * 3 + start,
                    WAVE_SIZE * 5 + start,
                    WAVE_SIZE * 5 + K + start,
                    WAVE_SIZE * 5 + K * 2 + start,
                    WAVE_SIZE * 5 + K * 3 + start,
                    WAVE_SIZE * 6 + start,
                    WAVE_SIZE * 6 + K + start,
                    WAVE_SIZE * 6 + K * 2 + start,
                    WAVE_SIZE * 6 + K * 3 + start,
                    WAVE_SIZE * 7 + start,
                    WAVE_SIZE * 7 + K + start,
                    WAVE_SIZE * 7 + K * 2 + start,
                    WAVE_SIZE * 7 + K * 3 + start,
                };
                return v;
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                using VecType = VecT<DataT, 1>;
                return VecType();
            }
        }
    };

    template <typename DataT, uint32_t VW, uint32_t K>
    ROCWMMA_DEVICE static inline bool aos_soa_b32()
    {
        bool err = false;

        const uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        auto           v         = AosVec<DataT, VW, K>::genData();

        __syncthreads();

        auto soa   = AosToSoa<K, VW>::exec(v);
        auto cmp_v = SoaVec<DataT, VW, K>::genData();
        err |= soa != cmp_v;

        return err;
    }

    template <typename DataT, uint32_t VW, uint32_t K>
    ROCWMMA_DEVICE static inline bool soa_aos_b32()
    {
        bool err = false;

        const uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        auto           v         = SoaVec<DataT, VW, K>::genData();
        __syncthreads();

        auto aos = SoaToAos<K, VW>::exec(v);

        auto cmp_v = AosVec<DataT, VW, K>::genData();
        err |= aos != cmp_v;

        return err;
    }

    template <typename DataT, uint32_t VW, uint32_t K>
    ROCWMMA_KERNEL void aossoaTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : aos_soa_b32<DataT, VW, K>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

    template <typename DataT, uint32_t VW, uint32_t K>
    ROCWMMA_KERNEL void soaaosTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : soa_aos_b32<DataT, VW, K>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
