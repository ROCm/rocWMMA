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

#ifndef ROCWMMA_SAMPLES_COMMON_HPP
#define ROCWMMA_SAMPLES_COMMON_HPP

#include <iostream>

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

bool checkCurrentDeviceIsgfx908()
{
    int deviceId = 0;
    CHECK_HIP_ERROR(hipGetDevice(&deviceId));
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, deviceId));
    std::string deviceName(prop.gcnArchName);
    bool is_gfx908 = false;
    if(deviceName.find("gfx908") != std::string::npos)
    {
        is_gfx908 = true;
    }
    return is_gfx908;
}

template <uint x>
struct Log2
{
    static constexpr uint value = 1 + Log2<x / 2>::value;
};

template <>
struct Log2<1>
{
    static constexpr uint value = 0;
};

// Matrix data initialization
template <typename DataT>
__host__ static inline void
    fill(DataT* mat, uint32_t m, uint32_t k, uint32_t b, uint32_t normalization = 1)
{
    auto batchOffset = m * k;
    for(int t = 0; t < b; ++t)
    {
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                // Random values normalized such that output is between 0 and 1
                auto value = __float2half(static_cast<float>(rand() / normalization)
                                          / static_cast<float>(RAND_MAX));
                mat[t * batchOffset + i * k + j] = static_cast<DataT>(value);
            }
        }
    }
}

// Element-wise comparison
template <typename T>
__host__ void compareEqual(T const* a, T const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval;
    double max_relative_error = 0.0;

    for(int i = 0; i < size; i++)
    {
        auto valA           = a[i];
        auto valB           = b[i];
        auto relative_error = fabs(valA - valB) / (fabs(valA) + fabs(valB) + 1.0);

        if(relative_error > max_relative_error || relative_error != relative_error)
        {
            max_relative_error = relative_error;
        }
    }
    auto eps = std::numeric_limits<T>::epsilon();
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAILED\n";
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << max_relative_error << std::endl;
}

#endif // ROCWMMA_SAMPLES_COMMON_HPP
