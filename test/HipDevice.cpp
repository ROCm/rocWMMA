#include "HipDevice.h"
#include "Common.hpp"

HipDevice::HipDevice()
    : mHandle(-1)
    , mGcnArch(hipGcnArch_t::UNKNOWN)
{
    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    mArch = mProps.arch;

    std::string deviceName(mProps.gcnArchName);

    if(deviceName.find("gfx908") != std::string::npos)
    {
        mGcnArch = hipGcnArch_t::GFX908;
    }
    else if(deviceName.find("gfx90a") != std::string::npos)
    {
        mGcnArch = hipGcnArch_t::GFX90A;
    }
}

hipDevice_t HipDevice::getDeviceHandle() const
{
    return mHandle;
}

hipDeviceProp_t HipDevice::getDeviceProps() const
{
    return mProps;
}

hipDeviceArch_t HipDevice::getDeviceArch() const
{
    return mArch;
}

HipDevice::hipGcnArch_t HipDevice::getGcnArch() const
{
    return mGcnArch;
}
