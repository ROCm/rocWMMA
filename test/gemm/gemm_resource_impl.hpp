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

#ifndef ROCWMMA_GEMM_RESOURCE_IMPL_HPP
#define ROCWMMA_GEMM_RESOURCE_IMPL_HPP

#include "gemm_resource.hpp"

namespace rocwmma
{

    template <typename InputT, typename OutputT>
    GemmResource<InputT, OutputT>::GemmResource()
        : mDeviceA(Base::template allocDevice<InputT>(0))
        , mDeviceB(Base::template allocDevice<InputT>(0))
        , mDeviceC(Base::template allocDevice<OutputT>(0))
        , mDeviceD(Base::template allocDevice<OutputT>(0))
        , mHostA(Base::template allocHost<InputT>(0))
        , mHostB(Base::template allocHost<InputT>(0))
        , mHostC(Base::template allocHost<OutputT>(0))
        , mHostD(Base::template allocHost<OutputT>(0))
        , mCurrentProblemSize({0, 0, 0})
        , mCurrentMatrixSize({0, 0, 0, 0})
        , mCurrentAllocSize({0, 0, 0, 0})
    {
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::copyHostToDeviceAll()
    {
        Base::copyData(mDeviceA, mHostA, std::get<MatrixA>(mCurrentMatrixSize));
        Base::copyData(mDeviceB, mHostB, std::get<MatrixB>(mCurrentMatrixSize));
        Base::copyData(mDeviceC, mHostC, std::get<MatrixC>(mCurrentMatrixSize));
        Base::copyData(mDeviceD, mHostD, std::get<MatrixD>(mCurrentMatrixSize));
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::copyDeviceToHostAll()
    {
        Base::copyData(mHostA, mDeviceA, std::get<MatrixA>(mCurrentMatrixSize));
        Base::copyData(mHostB, mDeviceB, std::get<MatrixB>(mCurrentMatrixSize));
        Base::copyData(mHostC, mDeviceC, std::get<MatrixC>(mCurrentMatrixSize));
        Base::copyData(mHostD, mDeviceD, std::get<MatrixD>(mCurrentMatrixSize));
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::resizeStorage(ProblemSize const& size)
    {
        auto calcMatrixSizes = [](ProblemSize const& size) {
            return std::make_tuple(std::get<M>(size) * std::get<K>(size), // sizeA = M * K
                                   std::get<K>(size) * std::get<N>(size), // sizeB = K * N
                                   std::get<M>(size) * std::get<N>(size), // sizeC = M * N
                                   std::get<M>(size) * std::get<N>(size)); // sizeD = M * N
        };

        auto allocIfNeeded =
            [](auto& devicePtr, auto& hostPtr, int64_t& currentAllocSize, int64_t newSize) {
                using DeviceDataT =
                    typename std::remove_reference_t<decltype(devicePtr)>::element_type;
                using HostDataT = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
                if(currentAllocSize < newSize)
                {
                    devicePtr = std::move(Base::template allocDevice<DeviceDataT>(newSize));
                    hostPtr   = std::move(Base::template allocHost<HostDataT>(newSize));

                    currentAllocSize = newSize;
                }
            };

        auto newSizes = calcMatrixSizes(size);

        allocIfNeeded(
            mDeviceA, mHostA, std::get<MatrixA>(mCurrentAllocSize), std::get<MatrixA>(newSizes));
        allocIfNeeded(
            mDeviceB, mHostB, std::get<MatrixB>(mCurrentAllocSize), std::get<MatrixB>(newSizes));
        allocIfNeeded(
            mDeviceC, mHostC, std::get<MatrixC>(mCurrentAllocSize), std::get<MatrixC>(newSizes));
        allocIfNeeded(
            mDeviceD, mHostD, std::get<MatrixD>(mCurrentAllocSize), std::get<MatrixD>(newSizes));

        mCurrentMatrixSize = newSizes;
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::reset()
    {
        mCurrentAllocSize  = {0, 0, 0, 0};
        mCurrentMatrixSize = {0, 0, 0, 0};

        auto allocNew = [](auto& devicePtr, auto& hostPtr) {
            using DeviceDataT = typename std::remove_reference_t<decltype(devicePtr)>::element_type;
            using HostDataT   = typename std::remove_reference_t<decltype(hostPtr)>::element_type;

            devicePtr = std::move(Base::template allocDevice<DeviceDataT>(0));
            hostPtr   = std::move(Base::template allocHost<HostDataT>(0));
        };

        allocNew(mDeviceA, mHostA);
        allocNew(mDeviceB, mHostB);
        allocNew(mDeviceC, mHostC);
        allocNew(mDeviceD, mHostD);
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

} // namespace rocwmma

#endif // ROCWMMA_GEMM_RESOURCE_IMPL_HPP
