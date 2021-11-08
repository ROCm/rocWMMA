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

#ifndef WMMA_UNIT_RESOURCE_H
#define WMMA_UNIT_RESOURCE_H

#include <memory>
#include <tuple>

#include <hip/hip_runtime_api.h>

#include "Singleton.h"

// UnitResource class is intended to manage a shared pool of resources for
// testing various kernels on the GPU.
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
struct UnitResource : public LazySingleton<UnitResource<DataT>>
{
    // For static initialization
    friend class LazySingleton<UnitResource<DataT>>;

    // M, N
    using ProblemSize = std::tuple<int64_t, int64_t>;

    // MatrixInOut(# of elements)
    using MatrixSize = int64_t;

    using DevicePtrT = std::unique_ptr<DataT, void (*)(DataT*)>;

    using HostPtrT = std::unique_ptr<DataT[]>;

    enum : uint32_t
    {
        // Problem size indices
        M = 0,
        N = 1
    };

public:
    UnitResource();
    ~UnitResource();

    static DevicePtrT allocDevice(int64_t numElements);

    static HostPtrT allocHost(int64_t numElements);

    void resizeStorage(ProblemSize const& size);

    static void copyData(HostPtrT& dst, DevicePtrT const& src, int64_t numElements);

    static void copyData(DevicePtrT& dst, HostPtrT const& src, int64_t numElements);

    HostPtrT& hostIn();

    DevicePtrT& deviceIn();
    DevicePtrT& deviceOut();

protected:
    DevicePtrT  mDeviceIn, mDeviceOut;
    HostPtrT    mHostIn;
    ProblemSize mCurrentProblemSize;
    MatrixSize  mCurrentMatrixSize;

private: // No copy
    UnitResource(UnitResource const&);
    UnitResource& operator=(UnitResource const&);
};

#include "UnitResource_impl.h"

#endif // WMMA_UNIT_RESOURCE_H
