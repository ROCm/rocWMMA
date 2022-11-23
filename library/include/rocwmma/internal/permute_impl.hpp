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
#ifndef ROCWMMA_PERMUTE_IMPL_HPP
#define ROCWMMA_PERMUTE_IMPL_HPP

#include "permute.hpp"

namespace rocwmma
{

    namespace detail
    {
        template <uint32_t WaveSize, uint32_t BlockSize, uint32_t BlockIdx>
        struct amdgcn_bpermute_block_bcast
        {
        private:
            enum Traits : uint32_t
            {
                BLOCK_OFFSET = BlockIdx * BlockSize,
            };

        public:
            // Calculate the read element based on thread position.
            __host__ __device__ static inline uint32_t threadCtrl(uint32_t threadId)
            {
                // Make sure that the threadId is within range
                auto tIdx = threadId % BlockSize;
                return Traits::BLOCK_OFFSET + tIdx;
            }
        };

        // bpermute: for the current thread, read from laneId
        struct amdgcn_ds_bpermute
        {
            template <typename InputT>
            __device__ static inline InputT exec(InputT input, uint32_t laneId)
            {
                // NOTE: final address is laneId * 4
                reinterpret_cast<uint32_t&>(input) = __builtin_amdgcn_ds_bpermute(
                    (laneId << 2), reinterpret_cast<uint32_t const&>(input));
                return input;
            }
        };

        // permute: for the current thread, push my value to laneId
        struct amdgcn_ds_permute
        {
            template <typename InputT>
            __device__ static inline InputT exec(InputT input, uint32_t laneId)
            {
                // NOTE: final address is laneId * 4
                reinterpret_cast<uint32_t&>(input) = __builtin_amdgcn_ds_permute(
                    (laneId << 2), reinterpret_cast<uint32_t const&>(input));
                return input;
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_IMPL_HPP
