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
    template <typename DataT, uint32_t VW, uint32_t K>
    struct AosVec;

    template <typename DataT, uint32_t VW, uint32_t K>
    struct SoaVec;

    template <typename DataT>
    struct AosVec<DataT, 4, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW = 4;
            using VecType         = VecT<DataT, VW>;
            VecType v             = {(uint8_t)threadIdx.x * VW,
                                     (uint8_t)threadIdx.x * VW + 1,
                                     (uint8_t)threadIdx.x * VW + 2,
                                     (uint8_t)threadIdx.x * VW + 3};
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
            VecType v             = {(uint8_t)threadIdx.x * VW,
                                     (uint8_t)threadIdx.x * VW + 1,
                                     (uint8_t)threadIdx.x * VW + 2,
                                     (uint8_t)threadIdx.x * VW + 3};
            return v;
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 4, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW = 4;
            using VecType         = VecT<DataT, VW>;
            VecType v             = {(uint8_t)threadIdx.x * VW,
                                     (uint8_t)threadIdx.x * VW + 1,
                                     (uint8_t)threadIdx.x * VW + 2,
                                     (uint8_t)threadIdx.x * VW + 3};
            return v;
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 4, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

            using VecType = VecT<DataT, VW * 2>;
            VecType v     = {(uint8_t)threadIdx.x * VW,
                             (uint8_t)threadIdx.x * VW + 1,
                             (uint8_t)threadIdx.x * VW + 2,
                             (uint8_t)threadIdx.x * VW + 3,
                             (uint8_t)threadIdx.x * VW + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 1 + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 2 + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 3 + VW * WAVE_SIZE};
            return v;
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 4, 256>
    {
        ROCWMMA_DEVICE static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;

            using VecType = VecT<DataT, VW * 4>;
            VecType v     = {(uint8_t)threadIdx.x * VW,
                             (uint8_t)threadIdx.x * VW + 1,
                             (uint8_t)threadIdx.x * VW + 2,
                             (uint8_t)threadIdx.x * VW + 3,
                             (uint8_t)threadIdx.x * VW + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 1 + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 2 + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + 3 + VW * WAVE_SIZE,
                             (uint8_t)threadIdx.x * VW + VW * WAVE_SIZE * 2,
                             (uint8_t)threadIdx.x * VW + 1 + VW * WAVE_SIZE * 2,
                             (uint8_t)threadIdx.x * VW + 2 + VW * WAVE_SIZE * 2,
                             (uint8_t)threadIdx.x * VW + 3 + VW * WAVE_SIZE * 2,
                             (uint8_t)threadIdx.x * VW + VW * WAVE_SIZE * 3,
                             (uint8_t)threadIdx.x * VW + 1 + VW * WAVE_SIZE * 3,
                             (uint8_t)threadIdx.x * VW + 2 + VW * WAVE_SIZE * 3,
                             (uint8_t)threadIdx.x * VW + 3 + VW * WAVE_SIZE * 3};
            return v;
        }
    };

    // SoaVec
    template <typename DataT>
    struct SoaVec<DataT, 4, 16>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 16;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            VecType v = {(uint8_t)threadIdx.x / K * K * 4 + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K * 2 + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K * 3 + ((uint8_t)threadIdx.x % K)};
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
            VecType v = {(uint8_t)threadIdx.x / K * K * 4 + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K * 2 + ((uint8_t)threadIdx.x % K),
                         (uint8_t)threadIdx.x / K * K * 4 + K * 3 + ((uint8_t)threadIdx.x % K)};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 64>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 64;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW>;
            VecType v                    = {(uint8_t)threadIdx.x,
                                            (uint8_t)threadIdx.x + K,
                                            (uint8_t)threadIdx.x + K * 2,
                                            (uint8_t)threadIdx.x + K * 3};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 128>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 128;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW * 2>;
            VecType v                    = {(uint8_t)threadIdx.x,
                                            (uint8_t)threadIdx.x + K,
                                            (uint8_t)threadIdx.x + K * 2,
                                            (uint8_t)threadIdx.x + K * 3,
                                            (uint8_t)threadIdx.x + WAVE_SIZE,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K * 2,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K * 3};
            return v;
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 256>
    {
        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            constexpr uint32_t VW        = 4;
            constexpr uint32_t K         = 256;
            constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
            using VecType                = VecT<DataT, VW * 4>;
            VecType v                    = {(uint8_t)threadIdx.x,
                                            (uint8_t)threadIdx.x + K,
                                            (uint8_t)threadIdx.x + K * 2,
                                            (uint8_t)threadIdx.x + K * 3,
                                            (uint8_t)threadIdx.x + WAVE_SIZE,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K * 2,
                                            (uint8_t)threadIdx.x + WAVE_SIZE + K * 3,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 2,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 2 + K,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 2 + K * 2,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 2 + K * 3,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 3,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 3 + K,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 3 + K * 2,
                                            (uint8_t)threadIdx.x + WAVE_SIZE * 3 + K * 3};
            return v;
        }
    };

    template <typename DataT, uint32_t VW, uint32_t K>
    ROCWMMA_DEVICE static inline bool aos_soa_b32()
    {
        bool err = false;

        const uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        auto           v         = AosVec<DataT, VW, K>::genData();
        __syncthreads();

        auto soa = AosToSoa<K, VW>::exec(v);

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
    ROCWMMA_KERNEL void transformsTest(uint32_t     m,
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
