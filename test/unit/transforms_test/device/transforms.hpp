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
            constexpr uint32_t VW = 8;
            using VecType         = VecT<DataT, VW>;
            auto    threadId      = (uint8_t)detail::threadId();
            VecType v             = {threadId * VW,
                                     threadId * VW + 1,
                                     threadId * VW + 2,
                                     threadId * VW + 3,
                                     threadId * VW + 4,
                                     threadId * VW + 5,
                                     threadId * VW + 6,
                                     threadId * VW + 7};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW = 8;
            using VecType         = VecT<DataT, VW>;
            auto    threadId      = (uint8_t)detail::threadId();
            VecType v             = {threadId * VW,
                                     threadId * VW + 1,
                                     threadId * VW + 2,
                                     threadId * VW + 3,
                                     threadId * VW + 4,
                                     threadId * VW + 5,
                                     threadId * VW + 6,
                                     threadId * VW + 7};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 8;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId * VW,
                                 threadId * VW + 1,
                                 threadId * VW + 2,
                                 threadId * VW + 3,
                                 threadId * VW + 4,
                                 threadId * VW + 5,
                                 threadId * VW + 6,
                                 threadId * VW + 7};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + 4,
                    threadId * VW + 5,
                    threadId * VW + 6,
                    threadId * VW + 7,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + 4 + VW * WAVE_SIZE,
                    threadId * VW + 5 + VW * WAVE_SIZE,
                    threadId * VW + 6 + VW * WAVE_SIZE,
                    threadId * VW + 7 + VW * WAVE_SIZE,
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
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + 4,
                    threadId * VW + 5,
                    threadId * VW + 6,
                    threadId * VW + 7,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + 4 + VW * WAVE_SIZE,
                    threadId * VW + 5 + VW * WAVE_SIZE,
                    threadId * VW + 6 + VW * WAVE_SIZE,
                    threadId * VW + 7 + VW * WAVE_SIZE,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + 4,
                    threadId * VW + 5,
                    threadId * VW + 6,
                    threadId * VW + 7,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + 4 + VW * WAVE_SIZE,
                    threadId * VW + 5 + VW * WAVE_SIZE,
                    threadId * VW + 6 + VW * WAVE_SIZE,
                    threadId * VW + 7 + VW * WAVE_SIZE,
                    threadId * VW + VW * WAVE_SIZE * 2,
                    threadId * VW + 1 + VW * WAVE_SIZE * 2,
                    threadId * VW + 2 + VW * WAVE_SIZE * 2,
                    threadId * VW + 3 + VW * WAVE_SIZE * 2,
                    threadId * VW + 4 + VW * WAVE_SIZE * 2,
                    threadId * VW + 5 + VW * WAVE_SIZE * 2,
                    threadId * VW + 6 + VW * WAVE_SIZE * 2,
                    threadId * VW + 7 + VW * WAVE_SIZE * 2,
                    threadId * VW + VW * WAVE_SIZE * 3,
                    threadId * VW + 1 + VW * WAVE_SIZE * 3,
                    threadId * VW + 2 + VW * WAVE_SIZE * 3,
                    threadId * VW + 3 + VW * WAVE_SIZE * 3,
                    threadId * VW + 4 + VW * WAVE_SIZE * 3,
                    threadId * VW + 5 + VW * WAVE_SIZE * 3,
                    threadId * VW + 6 + VW * WAVE_SIZE * 3,
                    threadId * VW + 7 + VW * WAVE_SIZE * 3,
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
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + 4,
                    threadId * VW + 5,
                    threadId * VW + 6,
                    threadId * VW + 7,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + 4 + VW * WAVE_SIZE,
                    threadId * VW + 5 + VW * WAVE_SIZE,
                    threadId * VW + 6 + VW * WAVE_SIZE,
                    threadId * VW + 7 + VW * WAVE_SIZE,
                    threadId * VW + VW * WAVE_SIZE * 2,
                    threadId * VW + 1 + VW * WAVE_SIZE * 2,
                    threadId * VW + 2 + VW * WAVE_SIZE * 2,
                    threadId * VW + 3 + VW * WAVE_SIZE * 2,
                    threadId * VW + 4 + VW * WAVE_SIZE * 2,
                    threadId * VW + 5 + VW * WAVE_SIZE * 2,
                    threadId * VW + 6 + VW * WAVE_SIZE * 2,
                    threadId * VW + 7 + VW * WAVE_SIZE * 2,
                    threadId * VW + VW * WAVE_SIZE * 3,
                    threadId * VW + 1 + VW * WAVE_SIZE * 3,
                    threadId * VW + 2 + VW * WAVE_SIZE * 3,
                    threadId * VW + 3 + VW * WAVE_SIZE * 3,
                    threadId * VW + 4 + VW * WAVE_SIZE * 3,
                    threadId * VW + 5 + VW * WAVE_SIZE * 3,
                    threadId * VW + 6 + VW * WAVE_SIZE * 3,
                    threadId * VW + 7 + VW * WAVE_SIZE * 3,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + 4,
                    threadId * VW + 5,
                    threadId * VW + 6,
                    threadId * VW + 7,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + 4 + VW * WAVE_SIZE,
                    threadId * VW + 5 + VW * WAVE_SIZE,
                    threadId * VW + 6 + VW * WAVE_SIZE,
                    threadId * VW + 7 + VW * WAVE_SIZE,
                    threadId * VW + VW * WAVE_SIZE * 2,
                    threadId * VW + 1 + VW * WAVE_SIZE * 2,
                    threadId * VW + 2 + VW * WAVE_SIZE * 2,
                    threadId * VW + 3 + VW * WAVE_SIZE * 2,
                    threadId * VW + 4 + VW * WAVE_SIZE * 2,
                    threadId * VW + 5 + VW * WAVE_SIZE * 2,
                    threadId * VW + 6 + VW * WAVE_SIZE * 2,
                    threadId * VW + 7 + VW * WAVE_SIZE * 2,
                    threadId * VW + VW * WAVE_SIZE * 3,
                    threadId * VW + 1 + VW * WAVE_SIZE * 3,
                    threadId * VW + 2 + VW * WAVE_SIZE * 3,
                    threadId * VW + 3 + VW * WAVE_SIZE * 3,
                    threadId * VW + 4 + VW * WAVE_SIZE * 3,
                    threadId * VW + 5 + VW * WAVE_SIZE * 3,
                    threadId * VW + 6 + VW * WAVE_SIZE * 3,
                    threadId * VW + 7 + VW * WAVE_SIZE * 3,
                    threadId * VW + VW * WAVE_SIZE * 4,
                    threadId * VW + 1 + VW * WAVE_SIZE * 4,
                    threadId * VW + 2 + VW * WAVE_SIZE * 4,
                    threadId * VW + 3 + VW * WAVE_SIZE * 4,
                    threadId * VW + 4 + VW * WAVE_SIZE * 4,
                    threadId * VW + 5 + VW * WAVE_SIZE * 4,
                    threadId * VW + 6 + VW * WAVE_SIZE * 4,
                    threadId * VW + 7 + VW * WAVE_SIZE * 4,
                    threadId * VW + VW * WAVE_SIZE * 5,
                    threadId * VW + 1 + VW * WAVE_SIZE * 5,
                    threadId * VW + 2 + VW * WAVE_SIZE * 5,
                    threadId * VW + 3 + VW * WAVE_SIZE * 5,
                    threadId * VW + 4 + VW * WAVE_SIZE * 5,
                    threadId * VW + 5 + VW * WAVE_SIZE * 5,
                    threadId * VW + 6 + VW * WAVE_SIZE * 5,
                    threadId * VW + 7 + VW * WAVE_SIZE * 5,
                    threadId * VW + VW * WAVE_SIZE * 6,
                    threadId * VW + 1 + VW * WAVE_SIZE * 6,
                    threadId * VW + 2 + VW * WAVE_SIZE * 6,
                    threadId * VW + 3 + VW * WAVE_SIZE * 6,
                    threadId * VW + 4 + VW * WAVE_SIZE * 6,
                    threadId * VW + 5 + VW * WAVE_SIZE * 6,
                    threadId * VW + 6 + VW * WAVE_SIZE * 6,
                    threadId * VW + 7 + VW * WAVE_SIZE * 6,
                    threadId * VW + VW * WAVE_SIZE * 7,
                    threadId * VW + 1 + VW * WAVE_SIZE * 7,
                    threadId * VW + 2 + VW * WAVE_SIZE * 7,
                    threadId * VW + 3 + VW * WAVE_SIZE * 7,
                    threadId * VW + 4 + VW * WAVE_SIZE * 7,
                    threadId * VW + 5 + VW * WAVE_SIZE * 7,
                    threadId * VW + 6 + VW * WAVE_SIZE * 7,
                    threadId * VW + 7 + VW * WAVE_SIZE * 7,
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
            constexpr uint32_t VW = 4;
            using VecType         = VecT<DataT, VW>;
            auto    threadId      = (uint8_t)detail::threadId();
            VecType v = {threadId * VW, threadId * VW + 1, threadId * VW + 2, threadId * VW + 3};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW = 4;
            using VecType         = VecT<DataT, VW>;
            auto    threadId      = (uint8_t)detail::threadId();
            VecType v = {threadId * VW, threadId * VW + 1, threadId * VW + 2, threadId * VW + 3};
            return v;
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW       = 4;
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v
                    = {threadId * VW, threadId * VW + 1, threadId * VW + 2, threadId * VW + 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 64 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId * VW,
                                 threadId * VW + 1,
                                 threadId * VW + 2,
                                 threadId * VW + 3,
                                 threadId * VW + VW * WAVE_SIZE,
                                 threadId * VW + 1 + VW * WAVE_SIZE,
                                 threadId * VW + 2 + VW * WAVE_SIZE,
                                 threadId * VW + 3 + VW * WAVE_SIZE};
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
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId * VW,
                                 threadId * VW + 1,
                                 threadId * VW + 2,
                                 threadId * VW + 3,
                                 threadId * VW + VW * WAVE_SIZE,
                                 threadId * VW + 1 + VW * WAVE_SIZE,
                                 threadId * VW + 2 + VW * WAVE_SIZE,
                                 threadId * VW + 3 + VW * WAVE_SIZE};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 128 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId * VW,
                                 threadId * VW + 1,
                                 threadId * VW + 2,
                                 threadId * VW + 3,
                                 threadId * VW + VW * WAVE_SIZE,
                                 threadId * VW + 1 + VW * WAVE_SIZE,
                                 threadId * VW + 2 + VW * WAVE_SIZE,
                                 threadId * VW + 3 + VW * WAVE_SIZE,
                                 threadId * VW + VW * WAVE_SIZE * 2,
                                 threadId * VW + 1 + VW * WAVE_SIZE * 2,
                                 threadId * VW + 2 + VW * WAVE_SIZE * 2,
                                 threadId * VW + 3 + VW * WAVE_SIZE * 2,
                                 threadId * VW + VW * WAVE_SIZE * 3,
                                 threadId * VW + 1 + VW * WAVE_SIZE * 3,
                                 threadId * VW + 2 + VW * WAVE_SIZE * 3,
                                 threadId * VW + 3 + VW * WAVE_SIZE * 3};
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
            auto               threadId = (uint8_t)detail::threadId();

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId * VW,
                                 threadId * VW + 1,
                                 threadId * VW + 2,
                                 threadId * VW + 3,
                                 threadId * VW + VW * WAVE_SIZE,
                                 threadId * VW + 1 + VW * WAVE_SIZE,
                                 threadId * VW + 2 + VW * WAVE_SIZE,
                                 threadId * VW + 3 + VW * WAVE_SIZE,
                                 threadId * VW + VW * WAVE_SIZE * 2,
                                 threadId * VW + 1 + VW * WAVE_SIZE * 2,
                                 threadId * VW + 2 + VW * WAVE_SIZE * 2,
                                 threadId * VW + 3 + VW * WAVE_SIZE * 2,
                                 threadId * VW + VW * WAVE_SIZE * 3,
                                 threadId * VW + 1 + VW * WAVE_SIZE * 3,
                                 threadId * VW + 2 + VW * WAVE_SIZE * 3,
                                 threadId * VW + 3 + VW * WAVE_SIZE * 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t VecSize   = VW * 256 / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId * VW,
                    threadId * VW + 1,
                    threadId * VW + 2,
                    threadId * VW + 3,
                    threadId * VW + VW * WAVE_SIZE,
                    threadId * VW + 1 + VW * WAVE_SIZE,
                    threadId * VW + 2 + VW * WAVE_SIZE,
                    threadId * VW + 3 + VW * WAVE_SIZE,
                    threadId * VW + VW * WAVE_SIZE * 2,
                    threadId * VW + 1 + VW * WAVE_SIZE * 2,
                    threadId * VW + 2 + VW * WAVE_SIZE * 2,
                    threadId * VW + 3 + VW * WAVE_SIZE * 2,
                    threadId * VW + VW * WAVE_SIZE * 3,
                    threadId * VW + 1 + VW * WAVE_SIZE * 3,
                    threadId * VW + 2 + VW * WAVE_SIZE * 3,
                    threadId * VW + 3 + VW * WAVE_SIZE * 3,
                    threadId * VW + VW * WAVE_SIZE * 4,
                    threadId * VW + 1 + VW * WAVE_SIZE * 4,
                    threadId * VW + 2 + VW * WAVE_SIZE * 4,
                    threadId * VW + 3 + VW * WAVE_SIZE * 4,
                    threadId * VW + VW * WAVE_SIZE * 5,
                    threadId * VW + 1 + VW * WAVE_SIZE * 5,
                    threadId * VW + 2 + VW * WAVE_SIZE * 5,
                    threadId * VW + 3 + VW * WAVE_SIZE * 5,
                    threadId * VW + VW * WAVE_SIZE * 6,
                    threadId * VW + 1 + VW * WAVE_SIZE * 6,
                    threadId * VW + 2 + VW * WAVE_SIZE * 6,
                    threadId * VW + 3 + VW * WAVE_SIZE * 6,
                    threadId * VW + VW * WAVE_SIZE * 7,
                    threadId * VW + 1 + VW * WAVE_SIZE * 7,
                    threadId * VW + 2 + VW * WAVE_SIZE * 7,
                    threadId * VW + 3 + VW * WAVE_SIZE * 7,
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
            constexpr uint32_t VW        = 8;
            constexpr uint32_t K         = 16;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            auto    threadId             = (uint8_t)detail::threadId();
            VecType v                    = {
                threadId / K * K * VW + threadId % K,
                threadId / K * K * VW + K + threadId % K,
                threadId / K * K * VW + K * 2 + threadId % K,
                threadId / K * K * VW + K * 3 + threadId % K,
                threadId / K * K * VW + K * 4 + threadId % K,
                threadId / K * K * VW + K * 5 + threadId % K,
                threadId / K * K * VW + K * 6 + threadId % K,
                threadId / K * K * VW + K * 7 + threadId % K,
            };
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 8;
            constexpr uint32_t K         = 32;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            auto    threadId             = (uint8_t)detail::threadId();
            VecType v                    = {
                threadId / K * K * VW + threadId % K,
                threadId / K * K * VW + K + threadId % K,
                threadId / K * K * VW + K * 2 + threadId % K,
                threadId / K * K * VW + K * 3 + threadId % K,
                threadId / K * K * VW + K * 4 + threadId % K,
                threadId / K * K * VW + K * 5 + threadId % K,
                threadId / K * K * VW + K * 6 + threadId % K,
                threadId / K * K * VW + K * 7 + threadId % K,
            };
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId / K * K * VW + threadId % K,
                    threadId / K * K * VW + K + threadId % K,
                    threadId / K * K * VW + K * 2 + threadId % K,
                    threadId / K * K * VW + K * 3 + threadId % K,
                    threadId / K * K * VW + K * 4 + threadId % K,
                    threadId / K * K * VW + K * 5 + threadId % K,
                    threadId / K * K * VW + K * 6 + threadId % K,
                    threadId / K * K * VW + K * 7 + threadId % K,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + K * 4,
                    threadId + K * 5,
                    threadId + K * 6,
                    threadId + K * 7,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE + K * 4,
                    threadId + WAVE_SIZE + K * 5,
                    threadId + WAVE_SIZE + K * 6,
                    threadId + WAVE_SIZE + K * 7,
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + K * 4,
                    threadId + K * 5,
                    threadId + K * 6,
                    threadId + K * 7,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE + K * 4,
                    threadId + WAVE_SIZE + K * 5,
                    threadId + WAVE_SIZE + K * 6,
                    threadId + WAVE_SIZE + K * 7,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + K * 4,
                    threadId + K * 5,
                    threadId + K * 6,
                    threadId + K * 7,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE + K * 4,
                    threadId + WAVE_SIZE + K * 5,
                    threadId + WAVE_SIZE + K * 6,
                    threadId + WAVE_SIZE + K * 7,
                    threadId + WAVE_SIZE * 2,
                    threadId + WAVE_SIZE * 2 + K,
                    threadId + WAVE_SIZE * 2 + K * 2,
                    threadId + WAVE_SIZE * 2 + K * 3,
                    threadId + WAVE_SIZE * 2 + K * 4,
                    threadId + WAVE_SIZE * 2 + K * 5,
                    threadId + WAVE_SIZE * 2 + K * 6,
                    threadId + WAVE_SIZE * 2 + K * 7,
                    threadId + WAVE_SIZE * 3,
                    threadId + WAVE_SIZE * 3 + K,
                    threadId + WAVE_SIZE * 3 + K * 2,
                    threadId + WAVE_SIZE * 3 + K * 3,
                    threadId + WAVE_SIZE * 3 + K * 4,
                    threadId + WAVE_SIZE * 3 + K * 5,
                    threadId + WAVE_SIZE * 3 + K * 6,
                    threadId + WAVE_SIZE * 3 + K * 7,
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + K * 4,
                    threadId + K * 5,
                    threadId + K * 6,
                    threadId + K * 7,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE + K * 4,
                    threadId + WAVE_SIZE + K * 5,
                    threadId + WAVE_SIZE + K * 6,
                    threadId + WAVE_SIZE + K * 7,
                    threadId + WAVE_SIZE * 2,
                    threadId + WAVE_SIZE * 2 + K,
                    threadId + WAVE_SIZE * 2 + K * 2,
                    threadId + WAVE_SIZE * 2 + K * 3,
                    threadId + WAVE_SIZE * 2 + K * 4,
                    threadId + WAVE_SIZE * 2 + K * 5,
                    threadId + WAVE_SIZE * 2 + K * 6,
                    threadId + WAVE_SIZE * 2 + K * 7,
                    threadId + WAVE_SIZE * 3,
                    threadId + WAVE_SIZE * 3 + K,
                    threadId + WAVE_SIZE * 3 + K * 2,
                    threadId + WAVE_SIZE * 3 + K * 3,
                    threadId + WAVE_SIZE * 3 + K * 4,
                    threadId + WAVE_SIZE * 3 + K * 5,
                    threadId + WAVE_SIZE * 3 + K * 6,
                    threadId + WAVE_SIZE * 3 + K * 7,
                };
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + K * 4,
                    threadId + K * 5,
                    threadId + K * 6,
                    threadId + K * 7,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE + K * 4,
                    threadId + WAVE_SIZE + K * 5,
                    threadId + WAVE_SIZE + K * 6,
                    threadId + WAVE_SIZE + K * 7,
                    threadId + WAVE_SIZE * 2,
                    threadId + WAVE_SIZE * 2 + K,
                    threadId + WAVE_SIZE * 2 + K * 2,
                    threadId + WAVE_SIZE * 2 + K * 3,
                    threadId + WAVE_SIZE * 2 + K * 4,
                    threadId + WAVE_SIZE * 2 + K * 5,
                    threadId + WAVE_SIZE * 2 + K * 6,
                    threadId + WAVE_SIZE * 2 + K * 7,
                    threadId + WAVE_SIZE * 3,
                    threadId + WAVE_SIZE * 3 + K,
                    threadId + WAVE_SIZE * 3 + K * 2,
                    threadId + WAVE_SIZE * 3 + K * 3,
                    threadId + WAVE_SIZE * 3 + K * 4,
                    threadId + WAVE_SIZE * 3 + K * 5,
                    threadId + WAVE_SIZE * 3 + K * 6,
                    threadId + WAVE_SIZE * 3 + K * 7,
                    threadId + WAVE_SIZE * 4,
                    threadId + WAVE_SIZE * 4 + K,
                    threadId + WAVE_SIZE * 4 + K * 2,
                    threadId + WAVE_SIZE * 4 + K * 3,
                    threadId + WAVE_SIZE * 4 + K * 4,
                    threadId + WAVE_SIZE * 4 + K * 5,
                    threadId + WAVE_SIZE * 4 + K * 6,
                    threadId + WAVE_SIZE * 4 + K * 7,
                    threadId + WAVE_SIZE * 5,
                    threadId + WAVE_SIZE * 5 + K,
                    threadId + WAVE_SIZE * 5 + K * 2,
                    threadId + WAVE_SIZE * 5 + K * 3,
                    threadId + WAVE_SIZE * 5 + K * 4,
                    threadId + WAVE_SIZE * 5 + K * 5,
                    threadId + WAVE_SIZE * 5 + K * 6,
                    threadId + WAVE_SIZE * 5 + K * 7,
                    threadId + WAVE_SIZE * 6,
                    threadId + WAVE_SIZE * 6 + K,
                    threadId + WAVE_SIZE * 6 + K * 2,
                    threadId + WAVE_SIZE * 6 + K * 3,
                    threadId + WAVE_SIZE * 6 + K * 4,
                    threadId + WAVE_SIZE * 6 + K * 5,
                    threadId + WAVE_SIZE * 6 + K * 6,
                    threadId + WAVE_SIZE * 6 + K * 7,
                    threadId + WAVE_SIZE * 7,
                    threadId + WAVE_SIZE * 7 + K,
                    threadId + WAVE_SIZE * 7 + K * 2,
                    threadId + WAVE_SIZE * 7 + K * 3,
                    threadId + WAVE_SIZE * 7 + K * 4,
                    threadId + WAVE_SIZE * 7 + K * 5,
                    threadId + WAVE_SIZE * 7 + K * 6,
                    threadId + WAVE_SIZE * 7 + K * 7,
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
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 16;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            auto    threadId             = (uint8_t)detail::threadId();
            VecType v                    = {threadId / K * K * 4 + threadId % K,
                                            threadId / K * K * 4 + K + threadId % K,
                                            threadId / K * K * 4 + K * 2 + threadId % K,
                                            threadId / K * K * 4 + K * 3 + threadId % K};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 32>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 32;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            auto    threadId             = (uint8_t)detail::threadId();
            VecType v                    = {threadId / K * K * 4 + threadId % K,
                                            threadId / K * K * 4 + K + threadId % K,
                                            threadId / K * K * 4 + K * 2 + threadId % K,
                                            threadId / K * K * 4 + K * 3 + threadId % K};
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId, threadId + K, threadId + K * 2, threadId + K * 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId,
                                 threadId + K,
                                 threadId + K * 2,
                                 threadId + K * 3,
                                 threadId + WAVE_SIZE,
                                 threadId + WAVE_SIZE + K,
                                 threadId + WAVE_SIZE + K * 2,
                                 threadId + WAVE_SIZE + K * 3};
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId,
                                 threadId + K,
                                 threadId + K * 2,
                                 threadId + K * 3,
                                 threadId + WAVE_SIZE,
                                 threadId + WAVE_SIZE + K,
                                 threadId + WAVE_SIZE + K * 2,
                                 threadId + WAVE_SIZE + K * 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId,
                                 threadId + K,
                                 threadId + K * 2,
                                 threadId + K * 3,
                                 threadId + WAVE_SIZE,
                                 threadId + WAVE_SIZE + K,
                                 threadId + WAVE_SIZE + K * 2,
                                 threadId + WAVE_SIZE + K * 3,
                                 threadId + WAVE_SIZE * 2,
                                 threadId + WAVE_SIZE * 2 + K,
                                 threadId + WAVE_SIZE * 2 + K * 2,
                                 threadId + WAVE_SIZE * 2 + K * 3,
                                 threadId + WAVE_SIZE * 3,
                                 threadId + WAVE_SIZE * 3 + K,
                                 threadId + WAVE_SIZE * 3 + K * 2,
                                 threadId + WAVE_SIZE * 3 + K * 3};
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

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {threadId,
                                 threadId + K,
                                 threadId + K * 2,
                                 threadId + K * 3,
                                 threadId + WAVE_SIZE,
                                 threadId + WAVE_SIZE + K,
                                 threadId + WAVE_SIZE + K * 2,
                                 threadId + WAVE_SIZE + K * 3,
                                 threadId + WAVE_SIZE * 2,
                                 threadId + WAVE_SIZE * 2 + K,
                                 threadId + WAVE_SIZE * 2 + K * 2,
                                 threadId + WAVE_SIZE * 2 + K * 3,
                                 threadId + WAVE_SIZE * 3,
                                 threadId + WAVE_SIZE * 3 + K,
                                 threadId + WAVE_SIZE * 3 + K * 2,
                                 threadId + WAVE_SIZE * 3 + K * 3};
                return v;
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t VecSize   = VW * K / Constants::AMDGCN_WAVE_SIZE;

                using VecType = VecT<DataT, VecSize>;
                VecType v     = {
                    threadId,
                    threadId + K,
                    threadId + K * 2,
                    threadId + K * 3,
                    threadId + WAVE_SIZE,
                    threadId + WAVE_SIZE + K,
                    threadId + WAVE_SIZE + K * 2,
                    threadId + WAVE_SIZE + K * 3,
                    threadId + WAVE_SIZE * 2,
                    threadId + WAVE_SIZE * 2 + K,
                    threadId + WAVE_SIZE * 2 + K * 2,
                    threadId + WAVE_SIZE * 2 + K * 3,
                    threadId + WAVE_SIZE * 3,
                    threadId + WAVE_SIZE * 3 + K,
                    threadId + WAVE_SIZE * 3 + K * 2,
                    threadId + WAVE_SIZE * 3 + K * 3,
                    threadId + WAVE_SIZE * 4,
                    threadId + WAVE_SIZE * 4 + K,
                    threadId + WAVE_SIZE * 4 + K * 2,
                    threadId + WAVE_SIZE * 4 + K * 3,
                    threadId + WAVE_SIZE * 5,
                    threadId + WAVE_SIZE * 5 + K,
                    threadId + WAVE_SIZE * 5 + K * 2,
                    threadId + WAVE_SIZE * 5 + K * 3,
                    threadId + WAVE_SIZE * 6,
                    threadId + WAVE_SIZE * 6 + K,
                    threadId + WAVE_SIZE * 6 + K * 2,
                    threadId + WAVE_SIZE * 6 + K * 3,
                    threadId + WAVE_SIZE * 7,
                    threadId + WAVE_SIZE * 7 + K,
                    threadId + WAVE_SIZE * 7 + K * 2,
                    threadId + WAVE_SIZE * 7 + K * 3,
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
