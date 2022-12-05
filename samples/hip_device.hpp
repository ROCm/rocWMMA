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

#ifndef ROCWMMA_SAMPLES_HIP_DEVICE_HPP
#define ROCWMMA_SAMPLES_HIP_DEVICE_HPP

namespace rocwmma
{
    enum hipWarpSize_t : uint32_t
    {
        Wave32 = 32,
        Wave64 = 64,
        UNSUPPORTED_WARP_SIZE,
    };

    uint32_t getWarpSize()
    {
        hipDevice_t     mHandle;
        hipDeviceProp_t mProps;
        uint32_t mWarpSize = hipWarpSize_t::UNSUPPORTED_WARP_SIZE;

        CHECK_HIP_ERROR(hipGetDevice(&mHandle));
        CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

        switch(mProps.warpSize)
        {
        case hipWarpSize_t::Wave32:
        case hipWarpSize_t::Wave64:
            mWarpSize = mProps.warpSize;
        default:;
        }

        if( mWarpSize == hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
        {
            std::cerr << "Cannot proceed: unsupported warp sizev detected. Exiting."
                        << std::endl;
            exit(EXIT_FAILURE);
        }

        return mWarpSize;
    }
} // namespace rocwmma

#endif // ROCWMMA_SAMPLES_HIP_DEVICE_HPP
