/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#ifndef DLRM_KERNEL_BASE_IMPL_H
#define DLRM_KERNEL_BASE_IMPL_H
#define __HIP_PLATFORM_HCC true

#include "DlrmKernelBase.h"
#include <cstdlib>
#include <iostream>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

#include "Common.hpp"
#include "Performance.h"

// Library includes
#include "Constants.h"
#include "Utils.h"

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::DlrmKernelBase()
{
    reset();
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::~DlrmKernelBase();
{
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::gridDim() const
{
    return dim3(ceilDiv(mM, BlockM * mTBlockX / AMDGCN_WAVE_SIZE), ceilDiv(mN, BlockN * mTBlockY));
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::blockDim() const
{
    return dim3(mTBlockX, mTBlockY);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::checkDevice() const
{
    auto deviceArch = DeviceInfo::instance()->getGcnArch();
    return (deviceArch != DeviceInfo::UNKNOWN
            && !(deviceArch == DeviceInfo::GFX908 && std::is_same<DataT, float64_t>::value));
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::checkSizes() const
{
    return (mM >= (BlockM * mTBlockX / AMDGCN_WAVE_SIZE) && mN >= (BlockN * mTBlockY)
            && mK >= BlockK);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::reset()
{
    mTBlockX = mTBlockY = 0;
    mM = mN = mK = 0;

    mRunFlag = true;

    mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
    mElapsedTimeMs = mEfficiency = 0.0;
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::setup(ProblemParams const& problem)
{
    // Reset the flags in case of multiple runs
    mRunFlag = true;

    // Format incoming problem parameters
    std::tie(mTBlockX, mTBlockY)
        = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                   static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
    std::tie(mM, mN, mK) = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<2>(problem.problemSize)));

    mRunFlag &= checkDevice();
    mRunFlag &= checkSizes();

    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();

        int         fp    = (std::is_same<DataT, float32_t>::value) ? 32 : 16;
        std::string fpStr = std::to_string(fp);

        // Determine whether to run forward or backward pass
        isBwd = problem.isBwd;

        // Initialize matrix storage
        if(!isBwd)
            dataInstance->resizeFwdStorage(problem.fwdDataSize);
        else
            dataInstance->resizeBwdStorage(problem.bwdDataSize);

        // Initialize matrix data on host and transfer to device
        // (check for null pointer for reading reference data only once)
        if(!isBwd)
        {
            int fdInput     = open("dlrmData/input_fp" + fpStr, O_RDONLY);
            int fdOutputRef = open("dlrmData/output_fp" + fpStr, O_RDONLY);

            pread(fdInput,
                  dataInstance->hostInput().get(),
                  std::get<0>(problem.fwdDataSize) * sizeof(DataT),
                  0);
            pread(fdOutputRef,
                  dataInstance->hostOutputRef().get(),
                  std::get<2>(problem.fwdDataSize) * sizeof(DataT),
                  0);

            dataInstance->copyHostToDeviceFwdAll();
        }
        else
        {
            int fdInput            = open("dlrmData/input_fp" + fpStr, O_RDONLY);
            int fdUpstreamGrad     = open("dlrmData/input_grad_fp" + fpStr, O_RDONLY);
            int fdGradRef          = open("dlrmData/output_input_grad_fp" + fpStr, O_RDONLY);
            int fdBottomMlpGradRef = open("dlrmData/output_mlp_input_grad_fp" + fpStr, O_RDONLY);

            pread(fdInput,
                  dataInstance->hostInput().get(),
                  std::get<0>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdUpstreamGrad,
                  dataInstance->hostUpstreamGrad().get(),
                  std::get<1>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdGradRef,
                  dataInstance->hostGradRef().get(),
                  std::get<3>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdBottomMlpGradRef,
                  dataInstance->hostBottomMlpGradRef().get(),
                  std::get<5>(problem.bwdDataSize) * sizeof(DataT),
                  0);

            dataInstance->copyHostToDeviceBwdAll();
        }
    }
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,

          typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::exec()
{
    if(mRunFlag)
    {
        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

        auto& dataInstance = DataStorage::instance();

        if(!isBwd)
        {
            hipExtLaunchKernelGGL((kernelFwdImpl()),
                                  (gridDim()),
                                  (blockDim()),
                                  (ldsUsage()),
                                  0,
                                  startEvent,
                                  stopEvent,
                                  0,
            // finish);
        }
        else
        {
            hipExtLaunchKernelGGL((kernelBwdImpl()),
                                  (gridDim()),
                                  (blockDim()),
                                  (ldsUsage()),
                                  0,
                                  startEvent,
                                  stopEvent,
                                  0,
            // finish);
        }

        // Calculate efficiency
        auto& deviceInfo             = DeviceInfo::instance();
        auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

        mElapsedTimeMs        = float64_t(timeMs);
        mTotalGFlops          = calculateGFlops(mM, mN, mK);
        mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs);
        mEfficiency           = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;
    }
}

#endif // DLRM_KERNEL_BASE_IMPL_H
