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

#ifndef DLRM_KERNEL_BASE_IMPL_HPP
#define DLRM_KERNEL_BASE_IMPL_HPP

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/utils.hpp>

#include "../common.hpp"
#include "./common.hpp"
#include "dlrm_kernel_base.hpp"
#include "performance.hpp"

// Library includes

#ifdef ROCWMMA_VALIDATION_TESTS
#include "reference.hpp" // Vanilla CPU kernel
#endif // ROCWMMA_VALIDATION_TESTS

namespace rocwmma
{

    template <uint32_t TileSize, typename DataT>
    DlrmKernelBase<TileSize, DataT>::DlrmKernelBase()
    {
        reset();
    }
    template <uint32_t TileSize, typename DataT>
    DlrmKernelBase<TileSize, DataT>::~DlrmKernelBase()
    {
    }

    template <uint32_t TileSize, typename DataT>
    uint32_t DlrmKernelBase<TileSize, DataT>::ldsUsage() const
    {
        return 0;
    }

    template <uint32_t TileSize, typename DataT>
    dim3 DlrmKernelBase<TileSize, DataT>::gridDim() const
    {
        if(passDirection == DlrmDirection_t::Forward)
        {
            return dim3(ceilDiv(mMPadded, TileSize * mTBlockX / AMDGCN_WAVE_SIZE),
                        ceilDiv(mM, TileSize),
                        mB);
        }
        else
        {
            return dim3(ceilDiv(mMPadded, TileSize * mTBlockX / AMDGCN_WAVE_SIZE),
                        ceilDiv(mK, TileSize),
                        mB);
        }
    }

    template <uint32_t TileSize, typename DataT>
    dim3 DlrmKernelBase<TileSize, DataT>::blockDim() const
    {
        return dim3(mTBlockX);
    }

    template <uint32_t TileSize, typename DataT>
    bool DlrmKernelBase<TileSize, DataT>::checkDevice() const
    {
        auto deviceArch = DeviceInfo::instance()->getGcnArch();
        return (deviceArch != DeviceInfo::UNKNOWN
                && !(deviceArch == DeviceInfo::GFX908 && std::is_same<DataT, float64_t>::value));
    }

    template <uint32_t TileSize, typename DataT>
    bool DlrmKernelBase<TileSize, DataT>::checkSizes() const
    {
        return (mM >= TileSize && (mM % TileSize == 0) && mK >= TileSize && (mK % TileSize == 0)
                && (mTBlockX % TileSize == 0));
    }

    template <uint32_t TileSize, typename DataT>
    bool DlrmKernelBase<TileSize, DataT>::checkLds() const
    {
        return ldsUsage() <= AMDGCN_LDS_MAX_SIZE_BYTES;
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::reset()
    {
        mM = mK = mB = 0;
        mMPadded = mKPadded = 0;
        mRepeats =
#ifdef ROCWMMA_VALIDATION_TESTS
            1;
#else
            5;
#endif

        mRunFlag = true;

        mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
        mElapsedTimeMs = mEfficiency = 0.0;

        passDirection = DlrmDirection_t::Forward;

        mValidationResult = false;
        mMaxRelativeError = 0.0;
    }

    template <uint32_t TileSize, typename DataT>
    HipResource* DlrmKernelBase<TileSize, DataT>::getResource()
    {
        return DataStorage::instance().get();
    }

    template <uint32_t TileSize, typename DataT>
    std::ostream& DlrmKernelBase<TileSize, DataT>::printHeader(std::ostream& stream) const
    {
        return stream << "TileSize, "
                      << "DataT, "
                      << "Direction, "
                      << "MatM, MatK, MatB, "
#if defined(ROCWMMA_VALIDATION_TESTS)
                      << "maxRelativeDiff, "
                      << "tolerance, "
#endif
                      << "elapsedMs, "
                      << "GFlops, "
                      << "GFlops/s, "
                      << "Efficiency(%)" << std::endl;
    }

    template <uint32_t TileSize, typename DataT>
    std::ostream& DlrmKernelBase<TileSize, DataT>::printKernel(std::ostream& stream) const
    {
        return stream << TileSize << ", " << dataTypeToString<DataT>() << ", "
                      << (passDirection == DlrmDirection_t::Forward ? "Forwards" : "Backwards")
                      << ", " << mM << ", " << mK << ", " << mB << ", "

#if defined(ROCWMMA_VALIDATION_TESTS)
                      << mMaxRelativeError << ", "
#endif
                      << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredGFlopsPerSec
                      << ", " << mEfficiency << ", "
#if defined(ROCWMMA_VALIDATION_TESTS)
                      << (mValidationResult ? "PASSED" : "FAILED")
#else
                      << "BENCH"
#endif // ROCWMMA_VALIDATION_TESTS
                      << std::endl;
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::setup(ProblemParams const& problem)
    {
        // Reset the flags in case of multiple runs
        mRunFlag = true;

        // Format incoming problem parameters
        std::tie(mTBlockX, mTBlockY)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                       static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
        std::tie(mM, mK, mB)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                       static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                       static_cast<uint32_t const&>(std::get<2>(problem.problemSize)));

        mMPadded = ceilDiv(mM, TileSize) * TileSize;
        mKPadded = ceilDiv(mK, TileSize) * TileSize;

        // Determine whether to run forward or backward pass
        passDirection = problem.passDirection;

        mRunFlag &= checkDevice();
        mRunFlag &= checkSizes();
        mRunFlag &= checkLds();

        if(mRunFlag)
        {
            auto& dataInstance = DataStorage::instance();

            // Initialize matrix storage
            if(passDirection == DlrmDirection_t::Forward)
            {
                dataInstance->resizeFwdStorage(problem.problemSize);
            }
            else
            {
                dataInstance->resizeBwdStorage(problem.problemSize);
            }

            // Initialize matrix data on device and transfer to host for validation
            if(passDirection == DlrmDirection_t::Forward)
            {
                MatrixUtil<row_major>::fillLaunchKernel(dataInstance->deviceInput().get(), mM, mK, mB);
#if defined(ROCWMMA_VALIDATION_TESTS)
                dataInstance->copyDeviceToHostFwdInput();
#endif // ROCWMMA_VALIDATION_TESTS
            }
            else
            {
                uint gradSize = ((mM * (mM - 1)) / 2) + mK;
                MatrixUtil<row_major>::fillLaunchKernel(dataInstance->deviceInput().get(), mM, mK, mB);
                MatrixUtil<row_major>::fillLaunchKernel(dataInstance->deviceUpstreamGrad().get(), 1, gradSize, mB);
#if defined(ROCWMMA_VALIDATION_TESTS)
                dataInstance->copyDeviceToHostBwdInput();
#endif // ROCWMMA_VALIDATION_TESTS
            }
        }
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::exec()
    {
        if(mRunFlag)
        {
            std::function<void()> dlrmKernel;
            if(passDirection == DlrmDirection_t::Forward)
            {
                if(mM == mMPadded && mK == mKPadded)
                {
                    uint inputBatchOffset  = mM * mK;
                    uint outputBatchOffset = ((mM * (mM - 1)) / 2) + mK;
                    uint accBatchOffset    = mM * mM;

                    dlrmKernel = [this, inputBatchOffset, outputBatchOffset, accBatchOffset]() {
                        auto& dataInstance = DataStorage::instance();
                        hipExtLaunchKernelGGL((this->kernelFwdImpl()),
                                              (this->gridDim()),
                                              (this->blockDim()),
                                              (this->ldsUsage()),
                                              0,
                                              nullptr,
                                              nullptr,
                                              0,
                                              dataInstance->deviceInput().get(),
                                              dataInstance->deviceOutput().get(),
                                              dataInstance->deviceAccFwd().get(),
                                              mM,
                                              mK,
                                              mB,
                                              inputBatchOffset,
                                              outputBatchOffset,
                                              accBatchOffset);
                    };
                }
            }
            else
            {
                if(mM == mMPadded && mK == mKPadded)
                {
                    auto& dataInstance = DataStorage::instance();

                    uint inputBatchOffset    = mM * mK;
                    uint upstreamBatchOffset = ((mM * (mM - 1)) / 2) + mK;
                    uint accBatchOffset      = mM * mM;

                    dlrmKernel = [this, inputBatchOffset, upstreamBatchOffset, accBatchOffset]() {
                        auto& dataInstance = DataStorage::instance();
                        auto  trilGridDim
                            = dim3(ceilDiv(mM * mM, static_cast<uint32_t>(mTBlockX)), 1, mB);

                        hipEvent_t syncEvent;
                        CHECK_HIP_ERROR(hipEventCreate(&syncEvent));
                        hipExtLaunchKernelGGL((this->kernelTrilImpl()),
                                              trilGridDim,
                                              this->blockDim(),
                                              0,
                                              0,
                                              nullptr,
                                              nullptr,
                                              0,
                                              dataInstance->deviceUpstreamGrad().get(),
                                              dataInstance->deviceAccBwd().get(),
                                              mM,
                                              mK,
                                              mB,
                                              upstreamBatchOffset,
                                              accBatchOffset);
                        CHECK_HIP_ERROR(hipEventRecord(syncEvent));
                        CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

                        hipExtLaunchKernelGGL((this->kernelBwdImpl()),
                                              (this->gridDim()),
                                              (this->blockDim()),
                                              (this->ldsUsage()),
                                              0,
                                              nullptr,
                                              nullptr,
                                              0,
                                              dataInstance->deviceInput().get(),
                                              dataInstance->deviceUpstreamGrad().get(),
                                              dataInstance->deviceGrad().get(),
                                              dataInstance->deviceBottomMlpGrad().get(),
                                              dataInstance->deviceAccBwd().get(),
                                              mM,
                                              mK,
                                              mB,
                                              inputBatchOffset,
                                              upstreamBatchOffset,
                                              accBatchOffset);
                    };
                }
            }

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

            CHECK_HIP_ERROR(hipEventRecord(startEvent));
            for(uint32_t i = 0; i < mRepeats; ++i)
            {
                dlrmKernel();
            }
            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            // Calculate efficiency
            auto& deviceInfo             = DeviceInfo::instance();
            auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<DataT>();
            auto  outputSize = (passDirection == DlrmDirection_t::Forward) ? mM * mM : mM * mK;

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = calculateGFlops(outputSize, mB, mK);
            mMeasuredGFlopsPerSec = calculateGFlopsPerSec(outputSize, mB, mK, mElapsedTimeMs)
                                    * static_cast<float64_t>(mRepeats);
            mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

#if defined(ROCWMMA_VALIDATION_TESTS)

            // Run reference CPU kernel
            std::function<void()> cpuKernel;
            if(passDirection == DlrmDirection_t::Forward)
            {
                cpuKernel = [this]() {
                    auto& dataInstance = DataStorage::instance();
                    dlrm_fwd_CPU<DataT>(dataInstance->hostInput().get(),
                                        dataInstance->hostOutputRef().get(),
                                        mM,
                                        mK,
                                        mB);
                };
            }
            else
            {
                cpuKernel = [this]() {
                    auto& dataInstance = DataStorage::instance();
                    dlrm_bwd_CPU<DataT>(dataInstance->hostInput().get(),
                                        dataInstance->hostUpstreamGrad().get(),
                                        dataInstance->hostBottomMlpGradRef().get(),
                                        dataInstance->hostGradRef().get(),
                                        mM,
                                        mK,
                                        mB);
                };
            }
            cpuKernel();
#endif
        }
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::validateResults()
    {
#ifdef ROCWMMA_VALIDATION_TESTS
        if(mRunFlag)
        {
            auto& dataInstance = DataStorage::instance();
            if(passDirection == DlrmDirection_t::Forward)
            {
                uint batchSize    = ((mM * (mM - 1)) / 2) + mK;
                auto reference = dataInstance->template allocDevice<DataT>(batchSize * mB);
                dataInstance->copyData(reference, dataInstance->hostOutputRef(), batchSize * mB);
                
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<DataT, DataT>(
                        dataInstance->deviceOutput().get(), reference.get(), 1, batchSize, mB, 10.0);

                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
            }
            else
            {
                // Copy reference output gradient to device
                auto reference0 = dataInstance->template allocDevice<DataT>(mM * mK * mB);
                dataInstance->copyData(reference0, dataInstance->hostGradRef(), mM * mK * mB);

                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<DataT, DataT>(
                        dataInstance->deviceGrad().get(), reference0.get(), mM, mK, mB);

                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;

                double maxRelativeError = mMaxRelativeError;
                
                // Copy reference bottom mlp gradient to device
                auto reference1 = dataInstance->template allocDevice<DataT>(mK * mB);
                dataInstance->copyData(reference1, dataInstance->hostBottomMlpGradRef(), mK * mB);

                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<DataT, DataT>(
                        dataInstance->deviceBottomMlpGrad().get(), reference1.get(), 1, mK, mB);

                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;

                mMaxRelativeError = (maxRelativeError > mMaxRelativeError) ? maxRelativeError : mMaxRelativeError;
            }
        }
#endif
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::reportResults()
    {
        if(!KernelI::sHeaderPrinted)
        {
            printHeader();
            KernelI::sHeaderPrinted = true;
        }
        printKernel();
    }

    template <uint32_t TileSize, typename DataT>
    void DlrmKernelBase<TileSize, DataT>::tearDown()
    {
    }

} // namespace rocwmma

#endif // DLRM_KERNEL_BASE_IMPL_HPP
