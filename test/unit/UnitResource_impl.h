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

#ifndef WMMA_UNIT_RESOURCE_IMPL_H
#define WMMA_UNIT_RESOURCE_IMPL_H

#include "Common.hpp"
#include <cstring> // for std::memcpy

template <typename DataT>
UnitResource<DataT>::UnitResource()
    : mDeviceIn(nullptr, [](DataT*) {})
    , mDeviceOut(nullptr, [](DataT*) {})
    , mHostIn(nullptr)
    , mCurrentProblemSize({0, 0})
    , mCurrentMatrixSize(0)
{
}

template <typename DataT>
auto UnitResource<DataT>::allocDevice(int64_t numElements) -> DevicePtrT
{
    DataT* data;
    CHECK_HIP_ERROR(hipMalloc(&data, numElements * sizeof(DataT)));
    return DevicePtrT(data, [](DataT* d) { CHECK_HIP_ERROR(hipFree(d)); });
}

template <typename DataT>
auto UnitResource<DataT>::allocHost(int64_t numElements) -> HostPtrT
{
    return HostPtrT(new DataT[numElements]);
}

template <typename DataT>
void UnitResource<DataT>::copyData(HostPtrT& dst, DevicePtrT const& src, int64_t numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyDeviceToHost));
}

template <typename DataT>
void UnitResource<DataT>::copyData(DevicePtrT& dst, HostPtrT const& src, int64_t numElements)
{
    CHECK_HIP_ERROR(
        hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyHostToDevice));
}

template <typename DataT>
void UnitResource<DataT>::resizeStorage(ProblemSize const& size)
{
    auto newSize = std::get<M>(size) * std::get<N>(size); // M * N = C, D

    if(mCurrentMatrixSize < newSize)
    {
        mHostIn    = std::move(allocHost(newSize));
        mDeviceIn  = std::move(allocDevice(newSize));
        mDeviceOut = std::move(allocDevice(newSize));
    }

    mCurrentMatrixSize = newSize;
}

template <typename DataT>
auto UnitResource<DataT>::hostIn() -> HostPtrT&
{
    return mHostIn;
}

template <typename DataT>
auto UnitResource<DataT>::deviceIn() -> DevicePtrT&
{
    return mDeviceIn;
}

template <typename DataT>
auto UnitResource<DataT>::deviceOut() -> DevicePtrT&
{
    return mDeviceOut;
}

#endif // WMMA_UNIT_RESOURCE_IMPL_H
