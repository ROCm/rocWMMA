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

#ifndef ROCWMMA_GEMM_RESOURCE_HPP
#define ROCWMMA_GEMM_RESOURCE_HPP

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

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

namespace rocwmma
{

    template <typename InputT, typename OutputT>
    struct GemmResource : public HipResource, public LazySingleton<GemmResource<InputT, OutputT>>
    {
        // For static initialization
        friend std::unique_ptr<GemmResource<InputT, OutputT>>
            std::make_unique<GemmResource<InputT, OutputT>>();

        using Base = HipResource;

        template <typename DataT>
        using DevicePtrT = Base::DevicePtrT<DataT>;

        template <typename DataT>
        using HostPtrT = Base::HostPtrT<DataT>;

        // M, N, K
        using ProblemSize = std::tuple<int64_t, int64_t, int64_t>;

        // MatrixA, MatrixB, MatrixCD (# of elements)
        using MatrixSize = std::tuple<int64_t, int64_t, int64_t>;

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

    protected:
        // Singleton instantiation
        GemmResource();
        GemmResource(GemmResource const&) = delete;
        GemmResource& operator=(GemmResource const&) = delete;

    public:
        ~GemmResource() = default;
        void copyHostToDeviceAll();
        void copyDeviceToHostAll();
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
    };

} // namespace rocwmma

#include "gemm_resource_impl.hpp"

#endif // ROCWMMA_GEMM_RESOURCE_HPP
