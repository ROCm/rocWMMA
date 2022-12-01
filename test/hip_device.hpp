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

#ifndef ROCWMMA_TEST_HIP_DEVICE_HPP
#define ROCWMMA_TEST_HIP_DEVICE_HPP

#include <hip/hip_runtime_api.h>

#include "performance.hpp"
#include "singleton.hpp"

namespace rocwmma
{

    class HipDevice : public LazySingleton<HipDevice>
    {
    public:
        // For static initialization
        friend std::unique_ptr<HipDevice> std::make_unique<HipDevice>();

        enum hipGcnArch_t : uint32_t
        {
            GFX908  = 0x908,
            GFX90A  = 0x90A,
            GFX1100 = 0x1100,
            GFX1101 = 0x1101,
            GFX1102 = 0x1102,
            UNSUPPORTED_ARCH,
        };

        enum hipWarpSize_t : uint32_t
        {
            Wave32 = 32,
            Wave64 = 64,
            UNSUPPORTED_WARP_SIZE,
        };

    protected:
        HipDevice();

    public:
        hipDevice_t     getDeviceHandle() const;
        hipDeviceProp_t getDeviceProps() const;
        hipDeviceArch_t getDeviceArch() const;
        hipGcnArch_t    getGcnArch() const;

        int warpSize() const;
        int sharedMemSize() const;
        int cuCount() const;
        int maxFreqMhz() const;

        template <typename InputT>
        double peakGFlopsPerSec() const;

    private:
        hipDevice_t     mHandle;
        hipDeviceProp_t mProps;
        hipDeviceArch_t mArch;
        hipGcnArch_t    mGcnArch;
        int             mWarpSize;
        int             mSharedMemSize;
        int             mCuCount;
        int             mMaxFreqMhz;
    };

    template <typename InputT>
    double HipDevice::peakGFlopsPerSec() const
    {
        double result = -1.0;
        switch(mGcnArch)
        {
        case hipGcnArch_t::GFX908:
            result = calculatePeakGFlopsPerSec<InputT, MI100>(mMaxFreqMhz, mCuCount);
            break;

        case hipGcnArch_t::GFX90A:
            result = calculatePeakGFlopsPerSec<InputT, MI200>(mMaxFreqMhz, mCuCount);
            break;
        default:;
        }
        return result;
    }

} // namespace rocwmma

#endif // ROCWMMA_TEST_HIP_DEVICE_HPP
