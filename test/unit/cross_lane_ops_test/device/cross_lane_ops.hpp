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
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Each thread operates on 32b or 64b data
        PackedT*       writeOut = reinterpret_cast<PackedT*>(out);
        PackedT const* readIn   = reinterpret_cast<PackedT const*>(in);
        PackedT        prev     = static_cast<PackedT>(param1);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset = blockIdx.x * blockDim.x + threadIdx.x;
        writeOut[dataOffset]
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                readIn[dataOffset], prev);
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
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Each thread operates on 32b or 64b data
        PackedT*       writeOut = reinterpret_cast<PackedT*>(out);
        PackedT const* readIn   = reinterpret_cast<PackedT const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset      = blockIdx.x * blockDim.x + threadIdx.x;
        writeOut[dataOffset] = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(readIn[dataOffset]);
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
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Each thread operates on 32b or 64b data
        PackedT*       writeOut = reinterpret_cast<PackedT*>(out);
        PackedT const* readIn   = reinterpret_cast<PackedT const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset      = blockIdx.x * blockDim.x + threadIdx.x;
        writeOut[dataOffset] = rocwmma::Permute::Driver<CrossLaneOp>::exec(readIn[dataOffset]);
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
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Each thread operates on 32b or 64b data
        PackedT*       writeOut = reinterpret_cast<PackedT*>(out);
        PackedT const* readIn   = reinterpret_cast<PackedT const*>(in);

        // Get offset into 1D array where all threads are neighbours.
        auto dataOffset = blockIdx.x * blockDim.x + threadIdx.x;
        writeOut[dataOffset]
            = rocwmma::Blend::Driver<CrossLaneOp>::exec(writeOut[dataOffset], readIn[dataOffset]);
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP
