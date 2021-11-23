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

#ifndef DLRM_RESOURCE_H
#define DLRM_RESOURCE_H

#include <memory>
#include <tuple>
//#include <hip/hip_runtime_api.h>
#include "HipResource.h"
#include "Singleton.h"

// DlrmResource class is intended to manage a shared pool of resources for
// testing DLRM kernels on the GPU.
//
// It minimizes the memory handling overhead for launching thousands of GPU
// kernels by allowing re-use of existing memory allocations. Memory is only
// re-allocated as necessary to satisfy minimum size requirements.
//
// The interface indicates memory ownership by this class and shall only be
// used to access for read/write purposes.
//
// Currently uses HIP as the backend for device allocation.

template <typename DataT>
struct DlrmResource : public HipResource, public LazySingleton<DlrmResource<DataT>>
{
    // For static initialization
    friend std::unique_ptr<DlrmResource<DataT>> std::make_unique<DlrmResource<DataT>>();

    using Base = HipResource;

    template <typename T>
    using DevicePtrT = Base::DevicePtrT<T>;

    template <typename T>
    using HostPtrT = Base::HostPtrT<T>;

    // M, N, K
    using ProblemSize = std::tuple<int64_t, int64_t, int64_t>;

    // Forward pass data sizes
    // Input, Output, OutputRef
    using DataSizeFwd = std::tuple<int64_t, int64_t, int64_t>;

    // Backward pass data sizes
    // Input, UpstreamGrad, Grad, GradRef, BottomMlpGrad, BottomMlpGradRef
    using DataSizeBwd = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

    enum : uint32_t
    {
        // Forward pass data size indices
        Input     = 0,
        Output    = 1,
        OutputRef = 2,

        // Backward pass data size indices
        UpstreamGrad     = 1,
        Grad             = 2,
        GradRef          = 3,
        BottomMlpGrad    = 4,
        BottomMlpGradRef = 5,

        // Problem size indices
        M = 0,
        N = 1,
        K = 2
    };

protected:
    // Singleton instantiation
    DlrmResource();
    DlrmResource(DlrmResource const&) = delete;
    DlrmResource& operator=(DlrmResource const&) = delete;

public:
    ~DlrmResource() = default;
    void copyHostToDeviceFwdAll();
    void copyHostToDeviceBwdAll();
    void resizeFwdStorage(DataSizeFwd const& size);
    void resizeBwdStorage(DataSizeBwd const& size);

    // Forward pass data
    HostPtrT<DataT>& hostInput();
    HostPtrT<DataT>& hostOutput();
    HostPtrT<DataT>& hostOutputRef();

    DevicePtrT<DataT>& deviceInput();
    DevicePtrT<DataT>& deviceOutput();
    DevicePtrT<DataT>& deviceOutputRef();

    // Backward pass data
    HostPtrT<DataT>& hostUpstreamGrad();
    HostPtrT<DataT>& hostGrad();
    HostPtrT<DataT>& hostGradRef();
    HostPtrT<DataT>& hostBottomMlpGrad();
    HostPtrT<DataT>& hostBottomMlpGradRef();

    DevicePtrT<DataT>& deviceUpstreamGrad();
    DevicePtrT<DataT>& deviceGrad();
    DevicePtrT<DataT>& deviceGradRef();
    DevicePtrT<DataT>& deviceBottomMlpGrad();
    DevicePtrT<DataT>& deviceBottomMlpGradRef();

    // Data sizes
    DataSizeFwd& currentDataSizeFwd();
    DataSizeBwd& currentDataSizeBwd();

protected:
    // Forward pass data
    DevicePtrT<DataT> mDeviceInput, mDeviceOutput, mDeviceOutputRef;
    HostPtrT<DataT>   mHostInput, mHostOutput, mHostOutputRef;

    // Backward pass data
    DevicePtrT<DataT> mDeviceUpstreamGrad, mDeviceGrad, mDeviceGradRef, mDeviceBottomMlpGrad,
        mDeviceBottomMlpGradRef;
    HostPtrT<DataT> mHostUpstreamGrad, mHostGrad, mHostGradRef, mHostBottomMlpGrad,
        mHostBottomMlpGradRef;

    ProblemSize mCurrentProblemSize;
    DataSizeFwd mCurrentDataSizeFwd;
    DataSizeBwd mCurrentDataSizeBwd;
};

#include "DlrmResource_impl.h"

#endif // DLRM_RESOURCE_H
