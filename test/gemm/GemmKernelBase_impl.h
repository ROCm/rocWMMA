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

#ifndef WMMA_KERNEL_BASE_IMPL_H
#define WMMA_KERNEL_BASE_IMPL_H

#include "GemmKernelBase.h"

#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

#include "Common.hpp"
#include "Performance.h"

#ifdef WMMA_VALIDATE_TESTS
#include "Reference.h" // Vanilla CPU kernel
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
#include "rocBLASReference.h" // rocBLAS GPU kernel
#endif // WMMA_VALIDATE_WITH_ROCBLAS
#endif // WMMA_VALIDATE_TESTS

// Library includes
#include "Constants.h"
#include "Utils.h"

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
GemmKernelBase<BlockM,
               BlockN,
               BlockK,
               InputT,
               OutputT,
               ComputeT,
               LayoutA,
               LayoutB,
               LayoutC,
               LayoutD>::GemmKernelBase()
{
    reset();
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
GemmKernelBase<BlockM,
               BlockN,
               BlockK,
               InputT,
               OutputT,
               ComputeT,
               LayoutA,
               LayoutB,
               LayoutC,
               LayoutD>::~GemmKernelBase()
{
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
uint32_t GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::ldsUsage() const
{
    return 0;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
dim3 GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::gridDim() const
{
    return dim3(ceilDiv(mM, BlockM * mTBlockX / AMDGCN_WAVE_SIZE), ceilDiv(mN, BlockN * mTBlockY));
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
dim3 GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::blockDim() const
{
    return dim3(mTBlockX, mTBlockY);
}

// Kernel run checks. Virtual as different GEMM kernels have different requirements
// True = run test
// False = skip test
template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
bool GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::checkDevice() const
{
    auto deviceArch = DeviceInfo::instance()->getGcnArch();
    return (deviceArch != DeviceInfo::UNKNOWN
            && !(deviceArch == DeviceInfo::GFX908 && std::is_same<InputT, float64_t>::value));
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
bool GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::checkSizes() const
{
    return (mM >= (BlockM * mTBlockX / AMDGCN_WAVE_SIZE) && mN >= (BlockN * mTBlockY)
            && mK >= BlockK);
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
bool GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::checkLds() const
{
    return ldsUsage() <= LDS_MAX_BYTES;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
bool GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::checkQuirks() const
{
    // Historically, there have been some bugs that are elicited under certain conditions
    // and produce 'quirky' failures that are beyond WMMA's control.
    // E.g. Previous compiler issue with h16 unsanitized register packing.
    // We can choose to ignore the quirks and focus on WMMA specific failures here.
    return true;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::reset()
{
    mTBlockX = mTBlockY = 0;
    mM = mN = mK = 0;
    mLda = mLdb = mLdc = mLdd = 0;
    mAlpha = mBeta = ComputeT(0.0f);

    mRunFlag          = true;
    mValidationResult = false;
    mMaxRelativeError = 0.0;

    mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
    mElapsedTimeMs = mEfficiency = 0.0;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
std::ostream& GemmKernelBase<BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             OutputT,
                             ComputeT,
                             LayoutA,
                             LayoutB,
                             LayoutC,
                             LayoutD>::printHeader(std::ostream& stream /* = std::cout */) const
{

    return stream
           << "TBlkX, TBlkY, BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, "
              "ldd, LytA_LytB_LytC_LytD, Ti_To_Tc, elapsedMs, GFlops, GFlops/s, Efficiency(%), "
              "Result"
           << std::endl;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
std::ostream& GemmKernelBase<BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             OutputT,
                             ComputeT,
                             LayoutA,
                             LayoutB,
                             LayoutC,
                             LayoutD>::printKernel(std::ostream& stream /* = std::cout */) const
{
    stream << mTBlockX << ", " << mTBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
           << ", " << mM << ", " << mN << ", " << mK << ", " << mAlpha << ", " << mLda << ", "
           << mLdb << ", " << mBeta << ", " << mLdc << ", " << mLdd << ", "
           << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << "_"
           << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << "_"
           << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << "_"
           << (std::is_same<LayoutD, row_major>::value ? "R" : "C") << ", "
           << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
           << dataTypeToString<ComputeT>() << ", ";

    if(!mRunFlag)
    {
        stream << "n/a"
               << ", "
               << "n/a"
               << ", "
               << "n/a"
               << ", "
               << "n/a"
               << ", "
               << " SKIPPED" << std::endl;
    }
    else
    {
        stream << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredGFlopsPerSec << ", "
               << mEfficiency << ", "
#ifdef WMMA_VALIDATE_TESTS
               << (mValidationResult ? "PASSED" : "FAILED")
#else
               << "BENCH"
#endif // WMMA_VALIDATE_TESTS
               << std::endl;
    }

    return stream;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::setup(ProblemParams const& problem)
{
    // Reset the flags in case of multiple runs
    mRunFlag          = true;
    mValidationResult = false;

    // Format incoming problem parameters
    std::tie(mTBlockX, mTBlockY)
        = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                   static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
    std::tie(mM, mN, mK) = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<2>(problem.problemSize)));
    std::tie(mAlpha, mBeta) = std::tie((ComputeT const&)static_cast<ComputeT>(problem.alpha),
                                       (ComputeT const&)static_cast<ComputeT>(problem.beta));
    std::tie(mLda, mLdb, mLdc, mLdd)
        = std::tie((std::is_same<LayoutA, row_major>::value ? mK : mM),
                   (std::is_same<LayoutB, row_major>::value ? mN : mK),
                   (std::is_same<LayoutC, row_major>::value ? mN : mM),
                   (std::is_same<LayoutC, row_major>::value ? mN : mM));

    // Clear the kernel to run
    mRunFlag &= checkDevice();
    mRunFlag &= checkSizes();
    mRunFlag &= checkLds();
    mRunFlag &= checkQuirks();

    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();

        // Initialize matrix storage
        dataInstance->resizeStorage(problem.problemSize);

        // Initialize matrix data on host
        MatrixUtil<LayoutA>::fill(dataInstance->hostA().get(), mM, mK);
        MatrixUtil<LayoutB>::fill(dataInstance->hostB().get(), mK, mN);
        MatrixUtil<LayoutC>::fill(dataInstance->hostC().get(), mM, mN);
        MatrixUtil<LayoutD>::fill(
            dataInstance->hostD().get(), mM, mN, std::numeric_limits<OutputT>::signaling_NaN());

        // Send to device
        dataInstance->copyHostToDeviceAll();
    }
};

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::exec()
{
    if(mRunFlag)
    {
        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

        auto& dataInstance = DataStorage::instance();

        hipExtLaunchKernelGGL((kernelImpl()), // Kernel to launch
                              (gridDim()), // Wg grid size
                              (blockDim()), // Thread block size
                              (ldsUsage()), // sharedMemBytes
                              0, // stream
                              startEvent, // Event start
                              stopEvent, // event stop
                              0, // flags
                              mM, // M
                              mN, // N
                              mK, // K
                              dataInstance->deviceA().get(), // A*
                              dataInstance->deviceB().get(), // B*
                              dataInstance->deviceC().get(), // C*
                              dataInstance->deviceD().get(), // D*
                              mLda, // lda
                              mLdb, // ldb
                              mLdc, // ldc
                              mLdd, // ldd
                              mAlpha, // alpha
                              mBeta); // beta

        auto timeMs = 0.0f;
        CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
        CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));
        CHECK_HIP_ERROR(hipEventDestroy(startEvent));
        CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

        // Calculate efficiency
        auto& deviceInfo             = DeviceInfo::instance();
        auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

        mElapsedTimeMs        = float64_t(timeMs);
        mTotalGFlops          = calculateGFlops(mM, mN, mK);
        mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs);
        mEfficiency           = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;
    }
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::validateResults()
{

#ifdef WMMA_VALIDATE_TESTS
    if(mRunFlag)
    {
        bool  validated    = false;
        auto& dataInstance = DataStorage::instance();

        // Allocated managed memory for results on host
        const int64_t sizeD           = mM * mN;
        auto          kernelResult    = dataInstance->template allocHost<OutputT>(sizeD);
        auto          referenceResult = dataInstance->template allocHost<OutputT>(sizeD);

        // Cache current kernel result from device
        dataInstance->copyData(kernelResult, dataInstance->deviceD(), sizeD);

        // Give more error tolerance to ComputeT = fp16,
        // due to MFMA output is always fp32. We downcast the MFMA result to fp16, which
        // will introduce an error compared to native fp16 MAC. The tolerance would be a function
        // of max / min values and number of operations propagating the error.
        // Note that integer values between [-2048, 2048 ] are exactly representable by fp16,
        // and significant rounding errors occur thereafter to the nearest multiple of 2.
        // The input generator for GEMM uses integer values within a certain range, therefore
        // FMA operations will be very prone to significant errors.
        double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;

#ifdef WMMA_VALIDATE_WITH_ROCBLAS

        // Attempt a reference result with rocBLAS
        if(quirks::rocblas_supported<InputT, OutputT, ComputeT>::value)
        {
            // rocblas matrix C, D always in col_major
            MatrixUtil<col_major>::fill(dataInstance->hostC().get(), mM, mN);
            gemm_rocBLAS<InputT, OutputT, ComputeT, LayoutA, LayoutB>(mM,
                                                                      mN,
                                                                      mK,
                                                                      dataInstance->hostA().get(),
                                                                      dataInstance->hostB().get(),
                                                                      dataInstance->hostC().get(),
                                                                      referenceResult.get(),
                                                                      mAlpha,
                                                                      mBeta);

            std::tie(mValidationResult, mMaxRelativeError)
                = compareEqual<OutputT, OutputT, LayoutD, col_major>(
                    kernelResult.get(), referenceResult.get(), mM, mN, errorTolerance);

            validated = true;
        }

#endif // WMMA_VALIDATE_WITH_ROCBLAS

        // Fall back to CPU validation if necessary
        if(!validated)
        {
            gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(
                mM,
                mN,
                mK,
                dataInstance->hostA().get(),
                dataInstance->hostB().get(),
                dataInstance->hostC().get(),
                referenceResult.get(),
                mAlpha,
                mBeta);

            std::tie(mValidationResult, mMaxRelativeError)
                = compareEqual<OutputT, OutputT, LayoutD, LayoutD>(
                    kernelResult.get(), referenceResult.get(), mM, mN, errorTolerance);
        }

        EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
    }
#endif // WMMA_VALIDATE_TESTS
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::reportResults()
{

    if(!KernelI::sHeaderPrinted)
    {
        printHeader();
        KernelI::sHeaderPrinted = true;
    }

    printKernel();
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void GemmKernelBase<BlockM,
                    BlockN,
                    BlockK,
                    InputT,
                    OutputT,
                    ComputeT,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    LayoutD>::tearDown()
{
}

#endif // WMMA_KERNEL_BASE_IMPL_H
