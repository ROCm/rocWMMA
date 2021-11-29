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

#ifndef DLRM_GEMM_RESOURCE_IMPL_H
#define DLRM_GEMM_RESOURCE_IMPL_H

#include "DlrmResource.h"

template <typename DataT>
DlrmResource<DataT>::DlrmResource()
    : mDeviceInput(Base::template allocDevice<DataT>(0))
    , mDeviceOutput(Base::template allocDevice<DataT>(0))
    , mDeviceOutputRef(Base::template allocDevice<DataT>(0))
    , mDeviceUpstreamGrad(Base::template allocDevice<DataT>(0))
    , mDeviceGrad(Base::template allocDevice<DataT>(0))
    , mDeviceGradRef(Base::template allocDevice<DataT>(0))
    , mDeviceBottomMlpGrad(Base::template allocDevice<DataT>(0))
    , mDeviceBottomMlpGradRef(Base::template allocDevice<DataT>(0))
    , mHostInput(Base::template allocHost<DataT>(0))
    , mHostOutput(Base::template allocHost<DataT>(0))
    , mHostOutputRef(Base::template allocHost<DataT>(0))
    , mHostUpstreamGrad(Base::template allocHost<DataT>(0))
    , mHostGrad(Base::template allocHost<DataT>(0))
    , mHostGradRef(Base::template allocHost<DataT>(0))
    , mHostBottomMlpGrad(Base::template allocHost<DataT>(0))
    , mHostBottomMlpGradRef(Base::template allocHost<DataT>(0))
    , mCurrentProblemSize({0, 0, 0})
    , mCurrentDataSizeFwd({0, 0, 0})
    , mCurrentDataSizeBwd({0, 0, 0, 0, 0, 0})
    , mMaxFwdCapacity({0, 0, 0})
    , mMaxBwdCapacity({0, 0, 0, 0, 0, 0})
{
}

template <typename DataT>
void DlrmResource<DataT>::copyHostToDeviceFwdAll()
{
    Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeFwd));
    Base::copyData(mDeviceOutput, mHostOutput, std::get<Output>(mCurrentDataSizeFwd));
    Base::copyData(mDeviceOutputRef, mHostOutputRef, std::get<OutputRef>(mCurrentDataSizeFwd));
}

template <typename DataT>
void DlrmResource<DataT>::copyHostToDeviceBwdAll()
{
    Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeBwd));
    Base::copyData(
        mDeviceUpstreamGrad, mHostUpstreamGrad, std::get<UpstreamGrad>(mCurrentDataSizeBwd));
    Base::copyData(mDeviceGrad, mHostGrad, std::get<Grad>(mCurrentDataSizeBwd));
    Base::copyData(mDeviceGradRef, mHostGradRef, std::get<GradRef>(mCurrentDataSizeBwd));
    Base::copyData(
        mDeviceBottomMlpGrad, mHostBottomMlpGrad, std::get<BottomMlpGrad>(mCurrentDataSizeBwd));
    Base::copyData(mDeviceBottomMlpGradRef,
                   mHostBottomMlpGradRef,
                   std::get<BottomMlpGradRef>(mCurrentDataSizeBwd));
}

template <typename DataT>
void DlrmResource<DataT>::resizeFwdStorage(DataSizeFwd const& size)
{
    auto calcMatrixSizes = [](DataSizeFwd const& size) {
        return std::make_tuple(
            std::get<Input>(size), std::get<Output>(size), std::get<OutputRef>(size));
    };

    auto allocIfNeeded = [](auto& devicePtr, auto& hostPtr, int64_t& currentMax, int64_t newSize) {
        using DeviceDataT = typename std::remove_reference_t<decltype(devicePtr)>::element_type;
        using HostDataT   = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
        if(currentMax < newSize)
        {
            currentMax = newSize;
            devicePtr  = std::move(Base::template allocDevice<DeviceDataT>(newSize));
            hostPtr    = std::move(Base::template allocHost<HostDataT>(newSize));
        }
    };

    auto newSizes = calcMatrixSizes(size);

    allocIfNeeded(
        mDeviceInput, mHostInput, std::get<Input>(mMaxFwdCapacity), std::get<Input>(newSizes));
    allocIfNeeded(
        mDeviceOutput, mHostOutput, std::get<Output>(mMaxFwdCapacity), std::get<Output>(newSizes));
    allocIfNeeded(mDeviceOutputRef,
                  mHostOutputRef,
                  std::get<OutputRef>(mMaxFwdCapacity),
                  std::get<OutputRef>(newSizes));

    mCurrentDataSizeFwd = newSizes;
}

template <typename DataT>
void DlrmResource<DataT>::resizeBwdStorage(DataSizeBwd const& size)
{
    auto calcMatrixSizes = [](DataSizeBwd const& size) {
        return std::make_tuple(std::get<Input>(size),
                               std::get<UpstreamGrad>(size),
                               std::get<Grad>(size),
                               std::get<GradRef>(size),
                               std::get<BottomMlpGrad>(size),
                               std::get<BottomMlpGradRef>(size));
    };

    auto allocIfNeeded = [](auto& devicePtr, auto& hostPtr, int64_t& currentMax, int64_t newSize) {
        using DeviceDataT = typename std::remove_reference_t<decltype(devicePtr)>::element_type;
        using HostDataT   = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
        if(currentMax < newSize)
        {
            currentMax = newSize;
            devicePtr  = std::move(Base::template allocDevice<DeviceDataT>(newSize));
            hostPtr    = std::move(Base::template allocHost<HostDataT>(newSize));
        }
    };

    auto newSizes = calcMatrixSizes(size);

    allocIfNeeded(
        mDeviceInput, mHostInput, std::get<Input>(mMaxBwdCapacity), std::get<Input>(newSizes));
    allocIfNeeded(mDeviceUpstreamGrad,
                  mHostUpstreamGrad,
                  std::get<UpstreamGrad>(mCurrentDataSizeBwd),
                  std::get<UpstreamGrad>(newSizes));
    allocIfNeeded(
        mDeviceGrad, mHostGrad, std::get<Grad>(mMaxBwdCapacity), std::get<Grad>(newSizes));
    allocIfNeeded(mDeviceGradRef,
                  mHostGradRef,
                  std::get<GradRef>(mMaxBwdCapacity),
                  std::get<GradRef>(newSizes));
    allocIfNeeded(mDeviceBottomMlpGrad,
                  mHostBottomMlpGrad,
                  std::get<BottomMlpGrad>(mMaxBwdCapacity),
                  std::get<BottomMlpGrad>(newSizes));
    allocIfNeeded(mDeviceBottomMlpGradRef,
                  mHostBottomMlpGradRef,
                  std::get<BottomMlpGradRef>(mMaxBwdCapacity),
                  std::get<BottomMlpGradRef>(newSizes));

    mCurrentDataSizeBwd = newSizes;
}

template <typename DataT>
auto DlrmResource<DataT>::hostInput() -> HostPtrT<DataT>&
{
    return mHostInput;
}

template <typename DataT>
auto DlrmResource<DataT>::hostOutput() -> HostPtrT<DataT>&
{
    return mHostOutput;
}

template <typename DataT>
auto DlrmResource<DataT>::hostOutputRef() -> HostPtrT<DataT>&
{
    return mHostOutputRef;
}

template <typename DataT>
auto DlrmResource<DataT>::hostUpstreamGrad() -> HostPtrT<DataT>&
{
    return mHostUpstreamGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::hostGrad() -> HostPtrT<DataT>&
{
    return mHostGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::hostGradRef() -> HostPtrT<DataT>&
{
    return mHostGradRef;
}

template <typename DataT>
auto DlrmResource<DataT>::hostBottomMlpGrad() -> HostPtrT<DataT>&
{
    return mHostBottomMlpGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::hostBottomMlpGradRef() -> HostPtrT<DataT>&
{
    return mHostBottomMlpGradRef;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceInput() -> DevicePtrT<DataT>&
{
    return mDeviceInput;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceOutput() -> DevicePtrT<DataT>&
{
    return mDeviceOutput;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceOutputRef() -> DevicePtrT<DataT>&
{
    return mDeviceOutputRef;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceUpstreamGrad() -> DevicePtrT<DataT>&
{
    return mDeviceUpstreamGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceGrad() -> DevicePtrT<DataT>&
{
    return mDeviceGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceGradRef() -> DevicePtrT<DataT>&
{
    return mDeviceGradRef;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceBottomMlpGrad() -> DevicePtrT<DataT>&
{
    return mDeviceBottomMlpGrad;
}

template <typename DataT>
auto DlrmResource<DataT>::deviceBottomMlpGradRef() -> DevicePtrT<DataT>&
{
    return mDeviceBottomMlpGradRef;
}

template <typename DataT>
auto DlrmResource<DataT>::currentDataSizeFwd() -> DataSizeFwd&
{
    return mCurrentDataSizeFwd;
}

template <typename DataT>
auto DlrmResource<DataT>::currentDataSizeBwd() -> DataSizeBwd&
{
    return mCurrentDataSizeBwd;
}

template <typename DataT>
auto DlrmResource<DataT>::maxFwdCapacity() -> DataSizeFwd&
{
    return mMaxFwdCapacity;
}

template <typename DataT>
auto DlrmResource<DataT>::maxBwdCapacity() -> DataSizeBwd&
{
    return mMaxBwdCapacity;
}
#endif // DLRM_GEMM_RESOURCE_IMPL_H
