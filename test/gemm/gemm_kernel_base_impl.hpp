/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_KERNEL_BASE_IMPL_HPP
#define ROCWMMA_KERNEL_BASE_IMPL_HPP

#include <cmath>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

// Library includes
#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/utils.hpp>

#include "common.hpp"
#include "gemm_kernel_base.hpp"
#include "performance.hpp"

#ifdef ROCWMMA_VALIDATION_TESTS
#include "reference.hpp" // Vanilla CPU kernel
#endif // ROCWMMA_VALIDATION_TESTS

#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS) || defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
#include "rocblas_reference.hpp" // rocBLAS GPU kernel
#endif // ROCWMMA_VALIDATE_WITH_ROCBLAS || ROCWMMA_BENCHMARK_WITH_ROCBLAS

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
        return dim3(ceilDiv(mM, BlockM * mTBlockX / DeviceInfo::instance()->warpSize()),
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
        auto& deviceInfo = DeviceInfo::instance();
        auto  deviceArch = deviceInfo->getGcnArch();

        // No unsupported devices
        return !(deviceArch == DeviceInfo::UNSUPPORTED_ARCH);
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
        // gridDim() takes the upper bound of block coverage.
        // In case of uneven division, this might put us out of bounds.
        // Forfeit the run because there is no tail for cleanup of remainders.
        auto tileSize = std::make_pair(BlockM * mTBlockX / DeviceInfo::instance()->warpSize(),
                                       BlockN * mTBlockY);
        auto gridDims = gridDim();
        return (gridDims.x * std::get<0>(tileSize) == mM)
               && (gridDims.y * std::get<1>(tileSize) == mN) && (mK % BlockK == 0) && BlockK <= mK;
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
        return ldsUsage() <= DeviceInfo::instance()->sharedMemSize();
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
        mTBlockX = mTBlockY = 0u;
        mM = mN = mK = 0u;
        mLda = mLdb = mLdc = mLdd = 0u;
        mAlpha = mBeta = static_cast<ComputeT>(0u);

        mRepeats =
#ifdef ROCWMMA_VALIDATION_TESTS
            1u;
#else
            5u;
#endif
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = 0.0;
        mEfficiency = mReferenceEfficiency = -1;
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
    template <template <uint32_t, uint32_t, uint32_t, uint32_t> class TestGuard>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::dispatchGuard()
    {

        // The test guard will be dispatched against 4 runtime params:
        // - TBlockX [32, 64, 128, 256]
        // - TBlockY [1, 2, 4]
        // - Wave Size [32, 64]
        // - Arch [gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1100, gfx1101, gfx1102]
        auto dispatchGuardFunc = [this]() {
            bool dispatchResult = false;

            auto waveSize   = DeviceInfo::instance()->warpSize();
            auto deviceArch = DeviceInfo::instance()->getGcnArch();

            // #define CASE_IMPL_ASSIGN4(TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
//     dispatchResult = TestGuard<TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID>::enable();

            // #define SWITCH_BODY_TBLOCK_X(TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
//     ROCWMMA_SWITCH_BODY4_ARG4(                             \
//         mTBlockX, CASE_IMPL_ASSIGN4, 32u, 64u, 128u, 256u, TBLOCK_Y, WAVE_SIZE, ARCH_ID)

            // #define SWITCH_BODY_TBLOCK_Y(WAVE_SIZE, ARCH_ID) \
//     ROCWMMA_SWITCH_BODY3_ARG3(mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, WAVE_SIZE, ARCH_ID)

            // #define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
//     ROCWMMA_SWITCH_BODY2_ARG2(         \
//         waveSize, SWITCH_BODY_TBLOCK_Y, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

            // #define DISPATCH_GUARD_BODY                          \
//     ROCWMMA_SWITCH_BODY8_ARG1(deviceArch,            \
//                               SWITCH_BODY_WAVE_SIZE, \
//                               HipDevice::GFX908,     \
//                               HipDevice::GFX90A,     \
//                               HipDevice::GFX940,     \
//                               HipDevice::GFX941,     \
//                               HipDevice::GFX942,     \
//                               HipDevice::GFX1100,    \
//                               HipDevice::GFX1101,    \
//                               HipDevice::GFX1102)

            //                 DISPATCH_GUARD_BODY

            // #undef CASE_IMPL_ASSIGN4
            // #undef SWITCH_BODY_TBLOCK_X
            // #undef SWITCH_BODY_TBLOCK_Y
            // #undef SWITCH_BODY_WAVE_SIZE
            // #undef DISPATCH_GUARD_BODY

            return dispatchResult;
        };

        // Finally, execute and return the dispatch guard result
        return dispatchGuardFunc();
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
    template <template <uint32_t, uint32_t, uint32_t, uint32_t> class KernelClass>
    auto GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::dispatchKernelFunc() -> KernelFunc
    {
        // The kernel function will be dispatched against 4 runtime params:
        // - TBlockX [32, 64, 128, 256]
        // - TBlockY [1, 2, 4]
        // - Wave Size [32, 64]
        // - Arch [gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1100, gfx1101, gfx1102]
        auto dispatchKernel = [this]() {
            auto waveSize   = DeviceInfo::instance()->warpSize();
            auto deviceArch = DeviceInfo::instance()->getGcnArch();

            // Runtime dispatcher to assign compile time TBlock params.
            auto result = typename decltype(*this)::KernelFunc(nullptr);

            // #define CASE_IMPL_ASSIGN4(TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID)  \
//         result = KernelClass<TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID>::generate();

            // #define SWITCH_BODY_TBLOCK_X(TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
// ROCWMMA_SWITCH_BODY4_ARG4(mTBlockX, CASE_IMPL_ASSIGN4, 32u, 64u, 128u, 256u, TBLOCK_Y, WAVE_SIZE, ARCH_ID)

            // #define SWITCH_BODY_TBLOCK_Y(WAVE_SIZE, ARCH_ID) \
// ROCWMMA_SWITCH_BODY3_ARG3(mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, WAVE_SIZE, ARCH_ID)

            // #define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
// ROCWMMA_SWITCH_BODY2_ARG2(waveSize, SWITCH_BODY_TBLOCK_Y, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

            // #define DISPATCH_KERNEL_FUNC_BODY \
// ROCWMMA_SWITCH_BODY8_ARG1(deviceArch, SWITCH_BODY_WAVE_SIZE, \
//                               HipDevice::GFX908,     \
//                               HipDevice::GFX90A,     \
//                               HipDevice::GFX940,     \
//                               HipDevice::GFX941,     \
//                               HipDevice::GFX942,     \
//                               HipDevice::GFX1100,    \
//                               HipDevice::GFX1101,    \
//                               HipDevice::GFX1102)

            //             DISPATCH_KERNEL_FUNC_BODY

            // #undef CASE_IMPL_ASSIGN4
            // #undef SWITCH_BODY_TBLOCK_X
            // #undef SWITCH_BODY_TBLOCK_Y
            // #undef SWITCH_BODY_WAVE_SIZE
            // #undef DISPATCH_KERNEL_FUNC_BODY

            return result;
        };

        return dispatchKernel();
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
    HipResource* GemmKernelBase<BlockM,
                                BlockN,
                                BlockK,
                                InputT,
                                OutputT,
                                ComputeT,
                                LayoutA,
                                LayoutB,
                                LayoutC,
                                LayoutD>::getResource() const
    {
        return DataStorage::instance().get();
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
                      << "Problem Size(GFlops), "
                      << "TFlops/s, "
                      << "Efficiency(%), "
#if defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
                      << "rocBLAS Efficiency(%), "
#endif // ROCWMMA_BENCHMARK_WITH_ROCBLAS
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
                                 LayoutD>::printKernel(std::ostream& stream) const
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
#if defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
                   << "n/a"
                   << ", "
#endif // ROCWMMA_BENCHMARK_WITH_ROCBLAS
                   << "SKIPPED" << std::endl;
        }
        else
        {

            stream << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredTFlopsPerSec
                   << ", " << mEfficiency << ", "
#if defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
                   << mReferenceEfficiency << ", "
#endif // ROCWMMA_BENCHMARK_WITH_ROCBLAS

#if defined(ROCWMMA_VALIDATION_TESTS)
                   << (mValidationResult ? "PASSED" : "FAILED")
#else
                   << "BENCH"
#endif // ROCWMMA_VALIDATION_TESTS
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

            // Initialize matrix data on device
            MatrixUtil<LayoutA>::fillLaunchKernel(dataInstance->deviceA().get(), mM, mK);
            MatrixUtil<LayoutB>::fillLaunchKernel(dataInstance->deviceB().get(), mK, mN);
            MatrixUtil<LayoutC>::fillLaunchKernel(dataInstance->deviceC().get(), mM, mN);
            MatrixUtil<LayoutD>::fillValLaunchKernel(dataInstance->deviceD().get(),
                                                     mM,
                                                     mN,
                                                     std::numeric_limits<OutputT>::signaling_NaN());

            // Copy to host if performing cpu validation
#if !defined(ROCWMMA_VALIDATE_WITH_ROCBLAS) && defined(ROCWMMA_VALIDATION_TESTS)
            dataInstance->copyDeviceToHostAll();
#endif // !defined(ROCWMMA_VALIDATE_WITH_ROCBLAS) && defined(ROCWMMA_VALIDATION_TESTS)

#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS)
            if(!quirks::rocblas_supported<InputT, OutputT, ComputeT>::value)
            {
                dataInstance->copyDeviceToHostAll();
            }
#endif // ROCWMMA_VALIDATE_WITH_ROCBLAS
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
                auto& deviceInfo = DeviceInfo::instance();

                auto devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredTFlopsPerSec = calculateTFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);

                mEfficiency = round(mMeasuredTFlopsPerSec / devicePeakGFlopsPerSec * 100000.0);

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }

            ///
            /// Select and run a reference kernel (if necessary)
            ///

            bool                  benchRef = false;
            std::function<void()> referenceKernel;

#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS) || defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)

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

                if constexpr((std::is_same<InputT, float8_t>::value)
                             || (std::is_same<InputT, bfloat8_t>::value))
                {
#if defined(ROCBLAS_DATA_TYPE_FLOAT8)
                    {
                        rocblas_computetype computeType = rocblas_compute_type_f32;
                        CHECK_ROCBLAS_ERROR(
                            rocblas_gemm_ex3(handle,
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
                                             computeType, // compute_type
                                             rocblas_gemm_algo_standard, // algo
                                             0, // solution_index
                                             0)); // flags
                    }
#endif
                }
                else
                {
                    CHECK_ROCBLAS_ERROR(
                        rocblas_gemm_ex(handle,
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
                }
            };
            if(quirks::rocblas_supported<InputT, OutputT, ComputeT>::value)
            {
                auto& dataInstance = DataStorage::instance();

                // A, B & C are already cached on GPU from ROCWMMA run.
                // Rocblas matrix C, D always in col_major, so we must
                // change C if needed
                if(!std::is_same<LayoutC, col_major>::value)
                {
                    MatrixUtil<col_major>::fillLaunchKernel(dataInstance->deviceC().get(), mM, mN);
                }

                benchRef        = true;
                referenceKernel = rocBlasKernel;

#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS)
                // Cache the ROCWMMA result from device
                auto rocwmmaResult = dataInstance->template allocHost<OutputT>(mM * mN);
                dataInstance->copyData(rocwmmaResult, dataInstance->deviceD(), mM * mN);

                // Reset device D with NaN
                MatrixUtil<LayoutD>::fillValLaunchKernel(
                    dataInstance->deviceD().get(),
                    mM,
                    mN,
                    std::numeric_limits<OutputT>::signaling_NaN());

                // Move the ROCWMMA result to host for analysis
                dataInstance->copyData(dataInstance->hostD(), rocwmmaResult, mM * mN);
#endif // ROCWMMA_VALIDATE_WITH_ROCBLAS
            }
#endif // ROCWMMA_VALIDATE_WITH_ROCBLAS || ROCWMMA_BENCHMARK_WITH_ROCBLAS

#if defined(ROCWMMA_VALIDATION_TESTS)

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
#endif // ROCWMMA_VALIDATION_TESTS

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
                    auto measuredTFlopsPerSec = calculateTFlopsPerSec(mM, mN, mK, elapsedTimeMs)
                                                * static_cast<float64_t>(mRepeats);
                    mReferenceEfficiency
                        = round(measuredTFlopsPerSec / devicePeakGFlopsPerSec * 100000.0);
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

#if defined(ROCWMMA_VALIDATION_TESTS)
        if(mRunFlag)
        {
            using DeviceLayoutD =
#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS)
                // rocBLAS output is col_major.
                typename std::conditional_t<
                    quirks::rocblas_supported<InputT, OutputT, ComputeT>::value,
                    col_major,
                    LayoutD>;
#else
                LayoutD;
#endif // ROCWMMA_VALIDATE_WITH_ROCBLAS

            auto& dataInstance = DataStorage::instance();

            // Allocated managed memory for results on host
            const int64_t sizeD = mM * mN;

            // One result on host needs to be transfered to device
            auto reference = dataInstance->template allocDevice<OutputT>(sizeD);
            dataInstance->copyData(reference, dataInstance->hostD(), sizeD);

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
                = compareEqualLaunchKernel<OutputT, OutputT, DeviceLayoutD, LayoutD>(
                    dataInstance->deviceD().get(), reference.get(), mM, mN, errorTolerance);

            // auto result = dataInstance->template allocHost<OutputT>(sizeD);
            // dataInstance->copyData(result, dataInstance->deviceD(), sizeD);

            // MatrixUtil<DeviceLayoutD>::print(dataInstance->hostD().get(), mM, mN);
            // MatrixUtil<LayoutD>::print(result.get(), mM, mN);

            EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
        }
#endif // ROCWMMA_VALIDATION_TESTS
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
                        LayoutD>::reportResults(std::ostream& stream,
                                                bool          omitHeader,
                                                bool          omitSkipped,
                                                bool          omitFailed,
                                                bool          omitPassed)
    {
        // Print header to std::cout
        if(!omitHeader)
        {
            printHeader(stream);
        }

        // Conditionally print kernel outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            printKernel(stream);
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
                        LayoutD>::tearDown()
    {
    }

} // namespace rocwmma

#endif // ROCWMMA_KERNEL_BASE_IMPL_HPP
