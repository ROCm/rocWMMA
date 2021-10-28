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

#include <cstring> // for std::memcpy

template <typename DataT>
DlrmResource<DataT>::DlrmResource()
    : mDeviceInput(nullptr, [](DataT*) {})
    , mDeviceOutput(nullptr, [](DataT*) {})
    , mDeviceOutputRef(nullptr, [](DataT*) {})
    , mDeviceUpstreamGrad(nullptr, [](DataT*) {})
    , mDeviceGrad(nullptr, [](DataT*) {})
    , mDeviceGradRef(nullptr, [](DataT*) {})
    , mDeviceBottomMlpGrad(nullptr, [](DataT*) {})
    , mDeviceBottomMlpGradRef(nullptr, [](DataT*) {})
    , mHostInput(nullptr)
    , mHostOutput(nullptr)
    , mHostOutputRef(nullptr)
    , mHostUpstreamGrad(nullptr)
    , mHostGrad(nullptr)
    , mHostGradRef(nullptr)
    , mHostBottomMlpGrad(nullptr)
    , mHostBottomMlpGradRef(nullptr)
    , mCurrentProblemSize({0, 0, 0})
    , mCurrentDataSizeFwd({0, 0, 0})
    , mCurrentDataSizeBwd({0, 0, 0, 0, 0, 0})
{
}

template <typename DataT>
DlrmResource<DataT>::~DlrmResource()
{
}

template <typename DataT>
auto DlrmResource<DataT>::allocDevice(int64_t numElements) -> DevicePtrT<DataT>
{
    DataT* data;
    CHECK_HIP_ERROR(hipMalloc(&data, numElements * sizeof(DataT)));
    return DevicePtrT<DataT>(data, [](DataT* d) { CHECK_HIP_ERROR(hipFree(d)); });
}

template <typename DataT>
auto DlrmResource<DataT>::allocHost(int64_t numElements) -> HostPtrT<DataT>
{
    return HostPtrT<DataT>(new DataT[numElements]);
}

template <typename DataT>
void DlrmResource<DataT>::copyData(HostPtrT<DataT>&         dst,
                                   DevicePtrT<DataT> const& src,
                                   int64_t                  numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyDeviceToHost));
}

template <typename DataT>
void DlrmResource<DataT>::copyData(DevicePtrT<DataT>&     dst,
                                   HostPtrT<DataT> const& src,
                                   int64_t                numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyHostToDevice));
}

template <typename DataT>
void DlrmResource<DataT>::copyData(HostPtrT<DataT>&       dst,
                                   HostPtrT<DataT> const& src,
                                   int64_t                numElements)
{
    std::memcpy(dst.get(), src.get(), numElements * sizeof(DataT));
}

template <typename DataT>
auto DlrmResource<DataT>::copyHostToDeviceFwdAll()
{
    copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeFwd));
    copyData(mDeviceOutput, mHostOutput, std::get<Output>(mCurrentDataSizeFwd));
    copyData(mDeviceOutputRef, mHostOutputRef, std::get<OutputRef>(mCurrentDataSizeFwd));
}

template <typename DataT>
auto DlrmResource<DataT>::copyHostToDeviceBwdAll()
{
    copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeFwd));
    copyData(mDeviceUpstreamGrad, mHostUpstreamGrad, std::get<UpstreamGrad>(mCurrentDataSizeFwd));
    copyData(mDeviceGrad, mHostGrad, std::get<Grad>(mCurrentDataSizeFwd));
    copyData(mDeviceGradRef, mHostGradRef, std::get<GradRef>(mCurrentDataSizeFwd));
    copyData(
        mDeviceBottomMlpGrad, mHostBottomMlpGrad, std::get<BottomMlpGrad>(mCurrentDataSizeFwd));
    copyData(mDeviceBottomMlpGradRef,
             mHostBottomMlpGradRef,
             std::get<BottomMlpGradRef>(mCurrentDataSizeFwd));
}

template <typename DataT>
auto DlrmResource<DataT>::resizeFwdStorage(DataSizeFwd const& size)
{
    auto calcMatrixSizes = [](DataSizeFwd const& size) {
        return std::make_tuple(
            std::get<Input>(size), std::get<Output>(size), std::get<OutputRef>(size));
    };

    auto allocIfNeeded = [](auto& devicePtr, auto& hostPtr, int64_t currentSize, int64_t newSize) {
        using DeviceDataT = typename std::remove_reference_t<decltype(devicePtr)>::element_type;
        using HostDataT   = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
        if(currentSize < newSize)
        {
            devicePtr = std::move(allocDevice<DeviceDataT>(newSize));
            hostPtr   = std::move(allocHost<HostDataT>(newSize));
        }
    };

    auto newSizes = calcMatrixSizes(size);

    allocIfNeeded(
        mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeFwd), std::get<Input>(newSizes));
    allocIfNeeded(mDeviceOutput,
                  mHostOutput,
                  std::get<Output>(mCurrentDataSizeFwd),
                  std::get<Output>(newSizes));
    allocIfNeeded(mDeviceOutputRef,
                  mHostOutputRef,
                  std::get<OutputRef>(mCurrentDataSizeFwd),
                  std::get<OutputRef>(newSizes));

    mCurrentDataSize = newSizes;
}

template <typename DataT>
auto DlrmResource<DataT>::resizeBwdStorage(DataSizeBwd const& size)
{
    auto calcMatrixSizes = [](DataSizeBwd const& size) {
        return std::make_tuple(std::get<Input>(size),
                               std::get<UpstreamGrad>(size),
                               std::get<Grad>(size) std::get<GradRef>(size)
                                   std::get<BottomMlpGrad>(size) std::get<BottomMlpGradRef>(size));
    };

    auto allocIfNeeded = [](auto& devicePtr, auto& hostPtr, int64_t currentSize, int64_t newSize) {
        using DeviceDataT = typename std::remove_reference_t<decltype(devicePtr)>::element_type;
        using HostDataT   = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
        if(currentSize < newSize)
        {
            devicePtr = std::move(allocDevice<DeviceDataT>(newSize));
            hostPtr   = std::move(allocHost<HostDataT>(newSize));
        }
    };

    auto newSizes = calcMatrixSizes(size);

    allocIfNeeded(
        mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeBwd), std::get<Input>(newSizes));
    allocIfNeeded(mDeviceUpstreamGrad,
                  mHostUpstreamGrad,
                  std::get<UpstreamGrad>(mCurrentDataSizeBwd),
                  std::get<UpstreamGrad>(newSizes));
    allocIfNeeded(
        mDeviceGrad, mHostGrad, std::get<Grad>(mCurrentDataSizeBwd), std::get<Grad>(newSizes));
    allocIfNeeded(mDeviceGradRef,
                  mHostGradRef,
                  std::get<GradRef>(mCurrentDataSizeBwd),
                  std::get<GradRef>(newSizes));
    allocIfNeeded(mDeviceBottomMlpGrad,
                  mHostBottomMlpGrad,
                  std::get<BottomMlpGrad>(mCurrentDataSizeBwd),
                  std::get<BottomMlpGrad>(newSizes));
    allocIfNeeded(mDeviceBottomMlpGradRef,
                  mHostBottomMlpGradRef,
                  std::get<BottomMlpGradRef>(mCurrentDataSizeBwd),
                  std::get<BottomMlpGradRef>(newSizes));

    mCurrentDataSize = newSizes;
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

#endif // DLRM_GEMM_RESOURCE_IMPL_H
