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

#ifndef WMMA_GEMM_RESOURCE_H
#define WMMA_GEMM_RESOURCE_H

#include <memory>
#include <tuple>

#include <hip/hip_runtime_api.h>

#include "Singleton.h"

// GemmResource class is intended to manage a shared pool of resources for
// testing GEMM kernels on the GPU.
//
// It minimizes the memory handling overhead for launching thousands of GPU
// kernels by allowing re-use of existing memory allocations. Memory is only
// re-allocated as necessary to satisfy minimum size requirements.
//
// The interface indicates memory ownership by this class and shall only be
// used to access for read/write purposes.
//
// Currently uses HIP as the backend for device allocation.

template <typename InputT, typename OutputT>
struct GemmResource : public LazySingleton<GemmResource<InputT, OutputT>>
{
    // For static initialization
    friend class LazySingleton<GemmResource<InputT, OutputT>>;

    // M, N, K
    using ProblemSize = std::tuple<int64_t, int64_t, int64_t>;

    // MatrixA, MatrixB, MatrixCD (# of elements)
    using MatrixSize = std::tuple<int64_t, int64_t, int64_t>;

    template <typename DataT>
    using DevicePtrT = std::unique_ptr<DataT, void (*)(DataT*)>;

    template <typename DataT>
    using HostPtrT = std::unique_ptr<DataT[]>;

    enum : uint32_t
    {
        // Matrix size indices
        MatrixA = 0,
        MatrixB = 1,
        MatrixC = 2,
        MatrixD = 2,

        // Problem size indices
        M = 0,
        N = 1,
        K = 2
    };

public:
    GemmResource();
    ~GemmResource();

    template <typename DataT>
    static DevicePtrT<DataT> allocDevice(int64_t numElements);

    template <typename DataT>
    static HostPtrT<DataT> allocHost(int64_t numElements);

    template <typename DataT>
    static void copyData(HostPtrT<DataT>& dst, DevicePtrT<DataT> const& src, int64_t numElements);

    template <typename DataT>
    static void copyData(DevicePtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);

    template <typename DataT>
    static void copyData(HostPtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);

    void copyHostToDeviceAll();
    void resizeStorage(ProblemSize const& size);

    HostPtrT<InputT>&  hostA();
    HostPtrT<InputT>&  hostB();
    HostPtrT<OutputT>& hostC();
    HostPtrT<OutputT>& hostD();

    DevicePtrT<InputT>&  deviceA();
    DevicePtrT<InputT>&  deviceB();
    DevicePtrT<OutputT>& deviceC();
    DevicePtrT<OutputT>& deviceD();

protected:
    DevicePtrT<InputT>  mDeviceA, mDeviceB;
    DevicePtrT<OutputT> mDeviceC, mDeviceD;
    HostPtrT<InputT>    mHostA, mHostB;
    HostPtrT<OutputT>   mHostC, mHostD;
    ProblemSize         mCurrentProblemSize;
    MatrixSize          mCurrentMatrixSize;

private: // No copy
    GemmResource(GemmResource const&);
    GemmResource& operator=(GemmResource const&);
};

#include "GemmResource_impl.h"

#endif // WMMA_GEMM_RESOURCE_H
