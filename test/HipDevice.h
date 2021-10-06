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
