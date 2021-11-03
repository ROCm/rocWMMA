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

#ifndef WMMA_HIP_RESOURCE_H
#define WMMA_HIP_RESOURCE_H

#include <memory>

// The HipResource class is intended as a wrapper for allocation, deletion and copying
// between host and device resources using the HIP backend.
// Memory is treated as a 1D array, and is managed through the std::unique_ptr class.

struct HipResource
{
    HipResource()                   = default;
    ~HipResource()                  = default;
    HipResource(const HipResource&) = delete;
    HipResource& operator=(const HipResource&) = delete;

    // Types
    template <typename DataT>
    using DevicePtrT = std::unique_ptr<DataT, void (*)(DataT*)>;

    template <typename DataT>
    using HostPtrT = std::unique_ptr<DataT[]>;

    // Alloc
    template <typename DataT>
    static DevicePtrT<DataT> allocDevice(int64_t numElements);

    template <typename DataT>
    static HostPtrT<DataT> allocHost(int64_t numElements);

    // Transfer wrappers
    template <typename DataT>
    static void copyData(HostPtrT<DataT>& dst, DevicePtrT<DataT> const& src, int64_t numElements);
    template <typename DataT>
    static void copyData(DevicePtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);
    template <typename DataT>
    static void copyData(HostPtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);
    template <typename DataT>
    static void copyData(DevicePtrT<DataT>& dst, DevicePtrT<DataT> const& src, int64_t numElements);
};

#include "HipResource_impl.h"

#endif // WMMA_HIP_RESOURCE_H
