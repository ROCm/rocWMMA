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

#ifndef ROCWMMA_KERNEL_BASE_IMPL_H
#define ROCWMMA_KERNEL_BASE_IMPL_H

#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

// Library includes
#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/utils.hpp>

#include "GemmKernelBase.h"
#include "common.hpp"
#include "performance.hpp"

#ifdef WMMA_VALIDATION_TESTS
#include "reference.hpp" // Vanilla CPU kernel
#endif // WMMA_VALIDATION_TESTS

#if defined(WMMA_VALIDATE_WITH_ROCBLAS) || defined(WMMA_BENCHMARK_WITH_ROCBLAS)
#include "rocblas_reference.hpp" // rocBLAS GPU kernel
#endif // WMMA_VALIDATE_WITH_ROCBLAS || WMMA_BENCHMARK_WITH_ROCBLAS

namespace rocwmma
{

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
        return dim3(ceilDiv(mM, BlockM * mTBlockX / rocwmma::AMDGCN_WAVE_SIZE),
                    ceilDiv(mN, BlockN * mTBlockY));
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
        return (mM >= (BlockM * mTBlockX / rocwmma::AMDGCN_WAVE_SIZE) && mN >= (BlockN * mTBlockY)
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
        return ldsUsage() <= rocwmma::AMDGCN_LDS_MAX_SIZE_BYTES;
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

        mRepeats =
#ifdef WMMA_VALIDATION_TESTS
            1;
#else
            5;
#endif
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
        mEfficiency = mReferenceEfficiency = -1.0;
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

        return stream << "TBlkX, TBlkY, "
                      << "BlkM, BlkN, BlkK, "
                      << "MatM, MatN, MatK, "
                      << "alpha, lda, ldb, beta, ldc, ldd, "
                      << "LytA_LytB_LytC_LytD, "
                      << "Ti_To_Tc, "
                      << "elapsedMs, "
                      << "GFlops, "
                      << "GFlops/s, "
                      << "Efficiency(%), "
#if defined(WMMA_BENCHMARK_WITH_ROCBLAS)
                      << "rocBLAS Efficiency(%), "
#endif // WMMA_BENCHMARK_WITH_ROCBLAS
                      << "Result" << std::endl;
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
               << dataTypeToString<LayoutA>() << "_" << dataTypeToString<LayoutB>() << "_"
               << dataTypeToString<LayoutC>() << "_" << dataTypeToString<LayoutD>() << ", "
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
#if defined(WMMA_BENCHMARK_WITH_ROCBLAS)
                   << "n/a"
                   << ", "
#endif // WMMA_BENCHMARK_WITH_ROCBLAS
                   << "SKIPPED" << std::endl;
        }
        else
        {

            stream << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredGFlopsPerSec
                   << ", " << mEfficiency << ", "
#if defined(WMMA_BENCHMARK_WITH_ROCBLAS)
                   << mReferenceEfficiency << ", "
#endif // WMMA_BENCHMARK_WITH_ROCBLAS

#if defined(WMMA_VALIDATION_TESTS)
                   << (mValidationResult ? "PASSED" : "FAILED")
#else
                   << "BENCH"
#endif // WMMA_VALIDATION_TESTS
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
        std::tie(mM, mN, mK)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
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
            ///
            /// Run ROCWMMA kernel
            ///

            auto rocwmmaKernel = [this]() {
                auto& dataInstance = DataStorage::instance();
                hipExtLaunchKernelGGL((this->kernelImpl()), // Kernel to launch
                                      (this->gridDim()), // Wg grid size
                                      (this->blockDim()), // Thread block size
                                      (this->ldsUsage()), // sharedMemBytes
                                      0, // stream
                                      nullptr, // Event start
                                      nullptr, // event stop
                                      0, // flags
                                      this->mM, // M
                                      this->mN, // N
                                      this->mK, // K
                                      dataInstance->deviceA().get(), // A*
                                      dataInstance->deviceB().get(), // B*
                                      dataInstance->deviceC().get(), // C*
                                      dataInstance->deviceD().get(), // D*
                                      this->mLda, // lda
                                      this->mLdb, // ldb
                                      this->mLdc, // ldc
                                      this->mLdd, // ldd
                                      this->mAlpha, // alpha
                                      this->mBeta); // beta
            };

            {
                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    rocwmmaKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                // Calculate efficiency
                auto& deviceInfo             = DeviceInfo::instance();
                auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);
                mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }

            ///
            /// Select and run a reference kernel (if necessary)
            ///

            bool                  benchRef = false;
            std::function<void()> referenceKernel;

#if defined(WMMA_VALIDATE_WITH_ROCBLAS) || defined(WMMA_BENCHMARK_WITH_ROCBLAS)

            // Create a rocBLAS handle to be used with rocBLAS API
            rocblas_handle handle;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

            // Create a guard object to release handle when it goes out of scope.
            using HandleGuardT = std::unique_ptr<rocblas_handle, void (*)(rocblas_handle*)>;
            auto handleGuard   = HandleGuardT(&handle, [](rocblas_handle* handle) {
                CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(*handle));
              });

            auto rocBlasKernel = [this, &handle]() {
                auto& dataInstance = DataStorage::instance();
                CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                                    rocblas_layout<LayoutA>::operation(), // opA
                                                    rocblas_layout<LayoutB>::operation(), // opB
                                                    this->mM, // M
                                                    this->mN, // N
                                                    this->mK, // K
                                                    &(this->mAlpha), // alpha,
                                                    dataInstance->deviceA().get(), // A*,
                                                    rocblas_types<InputT>::type(), // a_type
                                                    this->mLda, // lda
                                                    dataInstance->deviceB().get(), // B*,
                                                    rocblas_types<InputT>::type(), // b_type
                                                    this->mLdb, // ldb
                                                    &(this->mBeta), // beta
                                                    dataInstance->deviceC().get(), // C*
                                                    rocblas_types<OutputT>::type(), // c_type
                                                    this->mM, // ldc (col major output only)
                                                    dataInstance->deviceD().get(), // D*
                                                    rocblas_types<OutputT>::type(), // d_type
                                                    this->mM, // ldd (col major output only)
                                                    rocblas_types<ComputeT>::type(), // compute_type
                                                    rocblas_gemm_algo_standard, // algo
                                                    0, // solution_index
                                                    0)); // flags
            };
            if(quirks::rocblas_supported<InputT, OutputT, ComputeT>::value)
            {
                auto& dataInstance = DataStorage::instance();

                // A, B & C are already cached on GPU from ROCWMMA run.
                // Rocblas matrix C, D always in col_major, so we must
                // change C if needed
                if(!std::is_same<LayoutC, col_major>::value)
                {
                    MatrixUtil<col_major>::fill(dataInstance->hostC().get(), mM, mN);
                    dataInstance->copyData(dataInstance->deviceC(), dataInstance->hostC(), mM * mN);
                }

                benchRef        = true;
                referenceKernel = rocBlasKernel;

#if defined(WMMA_VALIDATE_WITH_ROCBLAS)
                // Cache the ROCWMMA result from device
                auto rocwmmaResult = dataInstance->template allocHost<OutputT>(mM * mN);
                dataInstance->copyData(rocwmmaResult, dataInstance->deviceD(), mM * mN);

                // Reset device D with NaN, from host
                dataInstance->copyData(dataInstance->deviceD(), dataInstance->hostD(), mM * mN);

                // Move the ROCWMMA result to host for analysis
                dataInstance->copyData(dataInstance->hostD(), rocwmmaResult, mM * mN);
#endif // WMMA_VALIDATE_WITH_ROCBLAS
            }
#endif // WMMA_VALIDATE_WITH_ROCBLAS || WMMA_BENCHMARK_WITH_ROCBLAS

#if defined(WMMA_VALIDATION_TESTS)

            // Fallback CPU kernel for validation
            auto cpuKernel = [this]() {
                auto& dataInstance = DataStorage::instance();
                gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(
                    this->mM,
                    this->mN,
                    this->mK,
                    dataInstance->hostA().get(),
                    dataInstance->hostB().get(),
                    dataInstance->hostC().get(),
                    dataInstance->hostD().get(), // Cpu result on host D
                    this->mAlpha,
                    this->mBeta);
            };

            if(!referenceKernel)
            {
                benchRef        = false; // No bench for cpu
                referenceKernel = cpuKernel;
            }
#endif // WMMA_VALIDATION_TESTS

            // Run reference kernel
            if(referenceKernel)
            {
                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    referenceKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                if(benchRef)
                {
                    // Calculate GPU efficiency
                    auto& deviceInfo             = DeviceInfo::instance();
                    auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

                    auto elapsedTimeMs        = float64_t(timeMs);
                    auto measuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, elapsedTimeMs)
                                                * static_cast<float64_t>(mRepeats);
                    mReferenceEfficiency = measuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;
                }

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }
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

#if defined(WMMA_VALIDATION_TESTS)
        if(mRunFlag)
        {
            using DeviceLayoutD =
#if defined(WMMA_VALIDATE_WITH_ROCBLAS)
                // rocBLAS output is col_major.
                typename std::conditional_t<
                    quirks::rocblas_supported<InputT, OutputT, ComputeT>::value,
                    col_major,
                    LayoutD>;
#else
                LayoutD;
#endif

            auto& dataInstance = DataStorage::instance();

            // Allocated managed memory for results on host
            const int64_t sizeD = mM * mN;

            // One result on device
            auto result0 = dataInstance->template allocHost<OutputT>(sizeD);
            dataInstance->copyData(result0, dataInstance->deviceD(), sizeD);

            // Other already on host
            auto& result1 = dataInstance->hostD();

            // Give more error tolerance to ComputeT = fp16,
            // due to MFMA output is always fp32. We downcast the MFMA result to fp16, which
            // will introduce an error compared to native fp16 MAC. The tolerance would be a function
            // of max / min values and number of operations propagating the error.
            // Note that integer values between [-2048, 2048 ] are exactly representable by fp16,
            // and significant rounding errors occur thereafter to the nearest multiple of 2.
            // The input generator for GEMM uses integer values within a certain range, therefore
            // FMA operations will be very prone to significant errors.
            double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;

            std::tie(mValidationResult, mMaxRelativeError)
                = compareEqual<OutputT, OutputT, DeviceLayoutD, LayoutD>(
                    result0.get(), result1.get(), mM, mN, errorTolerance);

            // MatrixUtil<DeviceLayoutD>::print(result0.get(), mM, mN);
            // MatrixUtil<LayoutD>::print(result1.get(), mM, mN);

            EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
        }
#endif // WMMA_VALIDATION_TESTS
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

} // namespace rocwmma

#endif // ROCWMMA_KERNEL_BASE_IMPL_H
