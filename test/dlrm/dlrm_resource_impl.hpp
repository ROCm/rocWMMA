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

#ifndef DLRM_GEMM_RESOURCE_IMPL_HPP
#define DLRM_GEMM_RESOURCE_IMPL_HPP

#include "dlrm_resource.hpp"

namespace rocwmma
{

    template <typename DataT>
    DlrmResource<DataT>::DlrmResource()
        : mDeviceInput(Base::template allocDevice<DataT>(0))
        , mDeviceOutput(Base::template allocDevice<DataT>(0))
        , mDeviceAccFwd(Base::template allocDevice<float>(0))
        , mDeviceUpstreamGrad(Base::template allocDevice<DataT>(0))
        , mDeviceGrad(Base::template allocDevice<DataT>(0))
        , mDeviceBottomMlpGrad(Base::template allocDevice<DataT>(0))
        , mDeviceAccBwd(Base::template allocDevice<DataT>(0))
        , mHostInput(Base::template allocHost<DataT>(0))
        , mHostOutput(Base::template allocHost<DataT>(0))
        , mHostOutputRef(Base::template allocHost<DataT>(0))
        , mHostAccFwd(Base::template allocHost<float>(0))
        , mHostUpstreamGrad(Base::template allocHost<DataT>(0))
        , mHostGrad(Base::template allocHost<DataT>(0))
        , mHostGradRef(Base::template allocHost<DataT>(0))
        , mHostBottomMlpGrad(Base::template allocHost<DataT>(0))
        , mHostBottomMlpGradRef(Base::template allocHost<DataT>(0))
        , mHostAccBwd(Base::template allocHost<DataT>(0))
        , mCurrentProblemSize({0, 0, 0})
        , mCurrentDataSizeFwd({0, 0, 0})
        , mCurrentDataSizeBwd({0, 0, 0, 0, 0})
        , mMaxFwdCapacity({0, 0, 0})
        , mMaxBwdCapacity({0, 0, 0, 0, 0})
    {
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyHostToDeviceFwdAll()
    {
        Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeFwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyHostToDeviceBwdAll()
    {
        Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentDataSizeBwd));
        Base::copyData(
            mDeviceUpstreamGrad, mHostUpstreamGrad, std::get<UpstreamGrad>(mCurrentDataSizeBwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostFwdOutput()
    {
        Base::copyData(mHostOutput, mDeviceOutput, std::get<Output>(mCurrentDataSizeFwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostBwdOutput()
    {
        Base::copyData(mHostGrad, mDeviceGrad, std::get<Grad>(mCurrentDataSizeBwd));
        Base::copyData(
            mHostBottomMlpGrad, mDeviceBottomMlpGrad, std::get<BottomMlpGrad>(mCurrentDataSizeBwd));
        Base::copyData(mHostAccBwd, mDeviceAccBwd, std::get<Acc>(mCurrentDataSizeBwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeFwdStorage(ProblemSize const& size)
    {
        auto calcTrilSize = [](ProblemSize const& size) {
            return ((std::get<M>(size) * (std::get<M>(size) - 1)) / 2) + std::get<K>(size);
        };

        auto calcMatrixSizes = [calcTrilSize](ProblemSize const& size) {
            return std::make_tuple(
                std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Input
                calcTrilSize(size) * std::get<B>(size), // Output
                std::get<M>(size) * std::get<M>(size) * std::get<B>(size)); // Acc
        };

        auto allocIfNeeded =
            [](auto& devicePtr, auto& hostPtr, int64_t& currentMax, int64_t newSize) {
                using DeviceDataT =
                    typename std::remove_reference_t<decltype(devicePtr)>::element_type;
                using HostDataT = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
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
        allocIfNeeded(mDeviceOutput,
                      mHostOutput,
                      std::get<Output>(mMaxFwdCapacity),
                      std::get<Output>(newSizes));
        allocIfNeeded(
            mDeviceAccFwd, mHostAccFwd, std::get<Acc>(mMaxFwdCapacity), std::get<Acc>(newSizes));
        mHostOutputRef = std::move(Base::template allocHost<DataT>(std::get<Output>(newSizes)));

        mCurrentDataSizeFwd = newSizes;
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeBwdStorage(ProblemSize const& size)
    {
        auto calcTrilSize = [](ProblemSize const& size) {
            return ((std::get<M>(size) * (std::get<M>(size) - 1)) / 2) + std::get<K>(size);
        };

        auto calcMatrixSizes = [calcTrilSize](ProblemSize const& size) {
            return std::make_tuple(
                std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Input
                calcTrilSize(size) * std::get<B>(size), // UpstreamGrad
                std::get<M>(size) * std::get<M>(size) * std::get<B>(size), // Acc
                std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Grad
                std::get<K>(size) * std::get<B>(size)); // BottomMlpGrad
        };

        auto allocIfNeeded =
            [](auto& devicePtr, auto& hostPtr, int64_t& currentMax, int64_t newSize) {
                using DeviceDataT =
                    typename std::remove_reference_t<decltype(devicePtr)>::element_type;
                using HostDataT = typename std::remove_reference_t<decltype(hostPtr)>::element_type;
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
        allocIfNeeded(mDeviceBottomMlpGrad,
                      mHostBottomMlpGrad,
                      std::get<BottomMlpGrad>(mMaxBwdCapacity),
                      std::get<BottomMlpGrad>(newSizes));
        allocIfNeeded(
            mDeviceAccBwd, mHostAccBwd, std::get<Acc>(mMaxBwdCapacity), std::get<Acc>(newSizes));
        mHostGradRef = std::move(Base::template allocHost<DataT>(std::get<Grad>(newSizes)));
        mHostBottomMlpGradRef
            = std::move(Base::template allocHost<DataT>(std::get<BottomMlpGrad>(newSizes)));

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
    auto DlrmResource<DataT>::hostAccFwd() -> HostPtrT<float>&
    {
        return mHostAccFwd;
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
    auto DlrmResource<DataT>::hostAccBwd() -> HostPtrT<DataT>&
    {
        return mHostAccBwd;
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
    auto DlrmResource<DataT>::deviceAccFwd() -> DevicePtrT<float>&
    {
        return mDeviceAccFwd;
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
    auto DlrmResource<DataT>::deviceBottomMlpGrad() -> DevicePtrT<DataT>&
    {
        return mDeviceBottomMlpGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceAccBwd() -> DevicePtrT<DataT>&
    {
        return mDeviceAccBwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::currentDataSizeFwd() const -> DataSizeFwd
    {
        return mCurrentDataSizeFwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::currentDataSizeBwd() const -> DataSizeBwd
    {
        return mCurrentDataSizeBwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::maxFwdCapacity() const -> DataSizeFwd
    {
        return mMaxFwdCapacity;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::maxBwdCapacity() const -> DataSizeBwd
    {
        return mMaxBwdCapacity;
    }

} // namespace rocwmma

#endif // DLRM_GEMM_RESOURCE_IMPL_HPP
