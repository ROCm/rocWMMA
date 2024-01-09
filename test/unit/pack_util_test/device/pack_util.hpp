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

#ifndef ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP
#define ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP

#include <rocwmma/rocwmma.hpp>

static constexpr uint32_t ERROR_VALUE   = 7u;
static constexpr uint32_t SUCCESS_VALUE = 0u;

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline DataT get(VecT<DataT, VecSize> const& v, uint32_t idx)
    {
        return v.data[idx];
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline auto generateSeqVec()
    {
        auto buildSeq = [](auto&& idx) {
            constexpr auto Index = std::decay_t<decltype(idx)>::value;
            return static_cast<DataT>(Index);
        };

        return vector_generator<DataT, VecSize>()(buildSeq);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool packUtilTestBasic()
    {
        bool err = false;

        auto res = generateSeqVec<DataT, VecSize>();

        for(uint32_t i = 0; i < VecSize; i++)
        {
            err |= (get(res, i) != static_cast<DataT>(i));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_KERNEL void packUtilTest(uint32_t     m,
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
        err = err ? err : packUtilTestBasic<DataT, VecSize>();

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

#endif // ROCWMMA_DEVICE_PACK_UTIL_TEST_HPP
