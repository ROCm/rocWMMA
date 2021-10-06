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

#ifndef WMMA_GEMM_RESOURCE_IMPL_H
#define WMMA_GEMM_RESOURCE_IMPL_H

#include <cstring> // for std::memcpy

template <typename InputT, typename OutputT>
GemmResource<InputT, OutputT>::GemmResource()
    : mDeviceA(nullptr, [](InputT*) {})
    , mDeviceB(nullptr, [](InputT*) {})
    , mDeviceC(nullptr, [](OutputT*) {})
    , mDeviceD(nullptr, [](OutputT*) {})
    , mHostA(nullptr)
    , mHostB(nullptr)
    , mHostC(nullptr)
    , mHostD(nullptr)
    , mCurrentProblemSize({0, 0, 0})
    , mCurrentMatrixSize({0, 0, 0})
{
}

template <typename InputT, typename OutputT>
GemmResource<InputT, OutputT>::~GemmResource()
{
}

template <typename InputT, typename OutputT>
template <typename DataT>
auto GemmResource<InputT, OutputT>::allocDevice(int64_t numElements) -> DevicePtrT<DataT>
{
    DataT* data;
    CHECK_HIP_ERROR(hipMalloc(&data, numElements * sizeof(DataT)));
    return DevicePtrT<DataT>(data, [](DataT* d) { CHECK_HIP_ERROR(hipFree(d)); });
}

template <typename InputT, typename OutputT>
template <typename DataT>
auto GemmResource<InputT, OutputT>::allocHost(int64_t numElements) -> HostPtrT<DataT>
{
    return HostPtrT<DataT>(new DataT[numElements]);
}

template <typename InputT, typename OutputT>
template <typename DataT>
void GemmResource<InputT, OutputT>::copyData(HostPtrT<DataT>&         dst,
                                             DevicePtrT<DataT> const& src,
                                             int64_t                  numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyDeviceToHost));
}

template <typename InputT, typename OutputT>
template <typename DataT>
void GemmResource<InputT, OutputT>::copyData(DevicePtrT<DataT>&     dst,
                                             HostPtrT<DataT> const& src,
                                             int64_t                numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyHostToDevice));
}

template <typename InputT, typename OutputT>
template <typename DataT>
void GemmResource<InputT, OutputT>::copyData(HostPtrT<DataT>&       dst,
                                             HostPtrT<DataT> const& src,
                                             int64_t                numElements)
{
    std::memcpy(dst.get(), src.get(), numElements * sizeof(DataT));
}

template <typename InputT, typename OutputT>
void GemmResource<InputT, OutputT>::copyHostToDeviceAll()
{
    copyData(mDeviceA, mHostA, std::get<MatrixA>(mCurrentMatrixSize));
    copyData(mDeviceB, mHostB, std::get<MatrixB>(mCurrentMatrixSize));
    copyData(mDeviceC, mHostC, std::get<MatrixC>(mCurrentMatrixSize));
    copyData(mDeviceD, mHostD, std::get<MatrixD>(mCurrentMatrixSize));
}

template <typename InputT, typename OutputT>
void GemmResource<InputT, OutputT>::resizeStorage(ProblemSize const& size)
{
    auto calcMatrixSizes = [](ProblemSize const& size) {
        return std::make_tuple(std::get<M>(size) * std::get<K>(size), // M * K = A
                               std::get<K>(size) * std::get<N>(size), // K * N = B
                               std::get<M>(size) * std::get<N>(size)); // M * N = C, D
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
        mDeviceA, mHostA, std::get<MatrixA>(mCurrentMatrixSize), std::get<MatrixA>(newSizes));
    allocIfNeeded(
        mDeviceB, mHostB, std::get<MatrixB>(mCurrentMatrixSize), std::get<MatrixB>(newSizes));
    allocIfNeeded(
        mDeviceC, mHostC, std::get<MatrixC>(mCurrentMatrixSize), std::get<MatrixC>(newSizes));
    allocIfNeeded(
        mDeviceD, mHostD, std::get<MatrixD>(mCurrentMatrixSize), std::get<MatrixD>(newSizes));

    mCurrentMatrixSize = newSizes;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::hostA() -> HostPtrT<InputT>&
{
    return mHostA;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::hostB() -> HostPtrT<InputT>&
{
    return mHostB;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::hostC() -> HostPtrT<OutputT>&
{
    return mHostC;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::hostD() -> HostPtrT<OutputT>&
{
    return mHostD;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::deviceA() -> DevicePtrT<InputT>&
{
    return mDeviceA;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::deviceB() -> DevicePtrT<InputT>&
{
    return mDeviceB;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::deviceC() -> DevicePtrT<OutputT>&
{
    return mDeviceC;
}

template <typename InputT, typename OutputT>
auto GemmResource<InputT, OutputT>::deviceD() -> DevicePtrT<OutputT>&
{
    return mDeviceD;
}

#endif // WMMA_GEMM_RESOURCE_IMPL_H
