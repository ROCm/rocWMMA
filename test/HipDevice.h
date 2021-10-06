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

#ifndef WMMA_TEST_HIP_DEVICE_H
#define WMMA_TEST_HIP_DEVICE_H

#include "Performance.h"
#include "Singleton.h"
#include <hip/hip_runtime_api.h>

class HipDevice : public LazySingleton<HipDevice>
{
public:
    // For static initialization
    friend class LazySingleton<HipDevice>;
    enum hipGcnArch_t : uint32_t
    {
        GFX908 = 0x908,
        GFX90A = 0x90A,
        UNKNOWN,
    };

public:
    HipDevice();

public:
    hipDevice_t     getDeviceHandle() const;
    hipDeviceProp_t getDeviceProps() const;
    hipDeviceArch_t getDeviceArch() const;
    hipGcnArch_t    getGcnArch() const;

    template <typename InputT>
    double peakGFlopsPerSec() const;

private:
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;
    hipDeviceArch_t mArch;
    hipGcnArch_t    mGcnArch;
};

template <typename InputT>
double HipDevice::peakGFlopsPerSec() const
{
    double result = -1.0;
    switch(mGcnArch)
    {
    case hipGcnArch_t::GFX908:
        result = calculatePeakGFlopsPerSec<InputT, MI100>(1087);
        break;

    case hipGcnArch_t::GFX90A:
        result = calculatePeakGFlopsPerSec<InputT, MI200>(985);
        break;
    default:;
    }
    return result;
}

#endif // WMMA_TEST_HIP_DEVICE_H
