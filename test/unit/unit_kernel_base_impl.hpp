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

#ifndef ROCWMMA_UNIT_KERNEL_BASE_IMPL_HPP
#define ROCWMMA_UNIT_KERNEL_BASE_IMPL_HPP

#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

// Library includes
#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/types.hpp>
#include <rocwmma/internal/utils.hpp>

#include "common.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    UnitKernelBase<BlockM, BlockN, DataT, Layout>::UnitKernelBase()
    {
        reset();
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    UnitKernelBase<BlockM, BlockN, DataT, Layout>::~UnitKernelBase()
    {
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    uint32_t UnitKernelBase<BlockM, BlockN, DataT, Layout>::ldsUsage() const
    {
        return 0u;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    dim3 UnitKernelBase<BlockM, BlockN, DataT, Layout>::gridDim() const
    {
        return dim3(ceilDiv(mM, BlockM * mTBlockX / AMDGCN_WAVE_SIZE),
                    ceilDiv(mN, BlockN * mTBlockY));
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    dim3 UnitKernelBase<BlockM, BlockN, DataT, Layout>::blockDim() const
    {
        return dim3(mTBlockX, mTBlockY);
    }

    // Kernel run checks. Virtual as different Test kernels have different requirements
    // True = run test
    // False = skip test
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    bool UnitKernelBase<BlockM, BlockN, DataT, Layout>::checkDevice() const
    {
        auto deviceArch = DeviceInfo::instance()->getGcnArch();
        return !(deviceArch == DeviceInfo::UNSUPPORTED
                || (deviceArch == DeviceInfo::GFX908 && std::is_same<DataT, float64_t>::value)
                || !(((deviceArch == DeviceInfo::GFX1100) || (deviceArch == DeviceInfo::GFX1101) || (deviceArch == DeviceInfo::GFX1102)) && (BlockM == 16) && (BlockN == 16)
                    && (std::is_same<DataT, float16_t>::value || std::is_same<DataT, bfloat16_t>::value || std::is_same<DataT, float32_t>::value
                    || std::is_same<DataT, int8_t>::value || std::is_same<DataT, int32_t>::value)));
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    bool UnitKernelBase<BlockM, BlockN, DataT, Layout>::checkSizes() const
    {
        // gridDim() takes the upper bound of block coverage.
        // In case of uneven division, this might put us out of bounds.
        // Forfeit the run because there is no tail for cleanup of remainders.
        auto tileSize = std::make_pair(BlockM * mTBlockX / AMDGCN_WAVE_SIZE, BlockN * mTBlockY);
        auto gridDims = gridDim();
        return (gridDims.x * std::get<0>(tileSize) == mM)
               && (gridDims.y * std::get<1>(tileSize) == mN);
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    bool UnitKernelBase<BlockM, BlockN, DataT, Layout>::checkLds() const
    {
        return ldsUsage() <= AMDGCN_LDS_MAX_SIZE_BYTES;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    bool UnitKernelBase<BlockM, BlockN, DataT, Layout>::checkQuirks() const
    {
        // Historically, there have been some bugs that are elicited under certain conditions
        // and produce 'quirky' failures that are beyond ROCWMMA's control.
        // E.g. Previous compiler issue with h16 unsanitized register packing.
        // We can choose to ignore the quirks and focus on ROCWMMA specific failures here.
        return true;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::reset()
    {
        mTBlockX = mTBlockY = 0;
        mM = mN = 0;
        mLd     = 0;
        mParam1 = mParam2 = DataT(0);

        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mTotalGFlops = mMeasuredTFlopsPerSec = 0.0;
        mElapsedTimeMs                       = 0.0;
        mEfficiency                          = 0.0;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    std::ostream& UnitKernelBase<BlockM, BlockN, DataT, Layout>::printHeader(
        std::ostream& stream /* = std::cout */) const
    {

        return stream << "TBlkX, TBlkY, BlkM, BlkN, MatM, MatN, Param1, ld, Param2, "
                         "Lyt, Td, elapsedMs, Problem Size(GFlops), TFlops/s, Efficiency(%), "
                         "Result"
                      << std::endl;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    std::ostream& UnitKernelBase<BlockM, BlockN, DataT, Layout>::printKernel(
        std::ostream& stream /* = std::cout */) const
    {
        stream << mTBlockX << ", " << mTBlockY << ", " << BlockM << ", " << BlockN << ", " << mM
               << ", " << mN << ", " << mLd << ", " << mParam1 << ", " << mParam2 << ", "
               << dataTypeToString<Layout>() << ", " << dataTypeToString<DataT>() << ", ";

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
            stream << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredTFlopsPerSec
                   << ", " << mEfficiency << ", " << (mValidationResult ? "PASSED" : "FAILED")
                   << std::endl;
        }

        return stream;
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::setup(ProblemParams const& problem)
    {
        // Reset the flags in case of multiple runs
        mRunFlag          = true;
        mValidationResult = false;

        // Format incoming problem parameters
        std::tie(mTBlockX, mTBlockY)
            = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                       static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
        std::tie(mM, mN) = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<1>(problem.problemSize)));
        mParam1          = static_cast<DataT>(problem.param1);
        mParam2          = static_cast<DataT>(problem.param2);
        mLd              = std::is_same<Layout, row_major>::value ? mN : mM;

        // Clear the kernel to run
        mRunFlag &= checkDevice();
        mRunFlag &= checkSizes();
        mRunFlag &= checkLds();
        mRunFlag &= checkQuirks();

        if(mRunFlag)
        {
            setupImpl(problem.problemSize);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::exec()
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
                                  dataInstance->deviceIn().get(), // In*
                                  dataInstance->deviceOut().get(), // Out*
                                  mLd, // ld
                                  mParam1, // param1
                                  mParam2); // param2

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));
            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            mElapsedTimeMs = float64_t(timeMs);
        }
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::validateResults()
    {
        if(mRunFlag)
        {
            validateResultsImpl();
        }
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::reportResults()
    {
        if(!KernelI::sHeaderPrinted)
        {
            printHeader();
            KernelI::sHeaderPrinted = true;
        }

        printKernel();
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    void UnitKernelBase<BlockM, BlockN, DataT, Layout>::tearDown()
    {
    }

} // namespace rocwmma

#endif // ROCWMMA_UNIT_KERNEL_BASE_IMPL_HPP
