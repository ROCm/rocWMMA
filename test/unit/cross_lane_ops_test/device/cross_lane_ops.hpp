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

#ifndef ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP
#define ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    __global__ void dppOpsTest(uint32_t     m,
                               uint32_t     n,
                               DataT const* in,
                               DataT*       out,
                               uint32_t     ld,
                               DataT        param1,
                               DataT        param2)
    {
        // Each thread operates on 32b data
        uint32_t*       write32Out = reinterpret_cast<uint32_t*>(out);
        uint32_t const* read32In   = reinterpret_cast<uint32_t const*>(in);
        uint32_t        prev       = static_cast<uint32_t>(param1);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset = blockIdx.x * blockDim.x + threadIdx.x;
        write32Out[dataOffset]
            = rocwmma::Dpp<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                read32In[dataOffset], prev);
    }

    template <typename DataT, typename CrossLaneOp>
    __global__ void swizzleOpsTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        // Each thread operates on 32b data
        uint32_t*       write32Out = reinterpret_cast<uint32_t*>(out);
        uint32_t const* read32In   = reinterpret_cast<uint32_t const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset        = blockIdx.x * blockDim.x + threadIdx.x;
        write32Out[dataOffset] = rocwmma::Swizzle<CrossLaneOp>::exec(read32In[dataOffset]);
    }

    template <typename DataT, typename CrossLaneOp>
    __global__ void permuteOpsTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        // Each thread operates on 32b data
        uint32_t*       write32Out = reinterpret_cast<uint32_t*>(out);
        uint32_t const* read32In   = reinterpret_cast<uint32_t const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset        = blockIdx.x * blockDim.x + threadIdx.x;
        write32Out[dataOffset] = rocwmma::Permute<CrossLaneOp>::exec(read32In[dataOffset]);
    }

    template <typename DataT, typename CrossLaneOp>
    __global__ void blendOpsTest(uint32_t     m,
                                 uint32_t     n,
                                 DataT const* in,
                                 DataT*       out,
                                 uint32_t     ld,
                                 DataT        param1,
                                 DataT        param2)
    {
        // Each thread operates on 32b data
        // Kernel uses out as src0 and in as src1, writing back to out
        uint32_t*       write32Out = reinterpret_cast<uint32_t*>(out);
        uint32_t const* read32In   = reinterpret_cast<uint32_t const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset = blockIdx.x * blockDim.x + threadIdx.x;
        write32Out[dataOffset]
            = rocwmma::Blend<CrossLaneOp>::exec(write32Out[dataOffset], read32In[dataOffset]);
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP
