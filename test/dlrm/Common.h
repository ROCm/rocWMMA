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

#ifndef DLRM_TEST_COMMON_H
#define DLRM_TEST_COMMON_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include "Constants.h"
#include "Types.h"
#include "Utils.h"
#include "WMMA.h"
#include "device/Common.h"

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

struct validate_data_t
{
    bool     pass;
    uint64_t numElements;
    float    maxAbsoluteDiff;
    float    maxRelativeDiff;
    float    tolerance = 1e-5;
};

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

struct __align__(8) half4
{
    half2 vals[2];
};

static void checkFileOpen(FILE* fp, std::string filename)
{
    if(fp == NULL)
    {
        std::cout << "Error opening: " << filename << "\n";
        exit(0);
    }
    return;
}

static bool checkFileSize(std::string filename, size_t fileSize, int64_t correctSize)
{
    if(fileSize != correctSize)
    {
        std::cout << "Error reading: " << filename << "\nRead size = " << fileSize
                  << ", Correct size = " << correctSize << "\n";

        return false;
    }
    return true;
}

template <typename T>
validate_data_t allclose(void* a, void* b, size_t bytes, bool verbose = false)
{
    size_t num_elm     = bytes / sizeof(T);
    size_t float_bytes = num_elm * sizeof(float);
    float *habs_diff, *hrel_diff;
    float *dabs_diff, *drel_diff;
    float *ha_float, *hb_float;
    float *da_float, *db_float;

    habs_diff = (float*)malloc(float_bytes);
    hrel_diff = (float*)malloc(float_bytes);
    ha_float  = (float*)malloc(float_bytes);
    hb_float  = (float*)malloc(float_bytes);
    hipMalloc(&dabs_diff, float_bytes);
    hipMalloc(&drel_diff, float_bytes);
    hipMalloc(&da_float, float_bytes);
    hipMalloc(&db_float, float_bytes);

    allclose_kernel<T, 1024>
        <<<1, 1024, 0>>>((T*)a, (T*)b, num_elm, dabs_diff, drel_diff, da_float, db_float);

    hipMemcpy(habs_diff, dabs_diff, float_bytes, hipMemcpyDefault);
    hipMemcpy(hrel_diff, drel_diff, float_bytes, hipMemcpyDefault);
    hipMemcpy(ha_float, da_float, float_bytes, hipMemcpyDefault);
    hipMemcpy(hb_float, db_float, float_bytes, hipMemcpyDefault);

    float      max_abs_diff = 0;
    float      max_rel_diff = 0;
    size_t     count        = 0;
    bool       failed       = false;
    const auto tolerance    = std::is_same<float, T>::value ? 1e-5 : 1e-2;
    for(size_t i = 0; i < num_elm; i++)
    {
        if(habs_diff[i] != 0)
        {
            count++;
            if(verbose)
            {
                std::cout << "[" << i << "] a " << ha_float[i] << ", b " << hb_float[i]
                          << ", abs diff " << habs_diff[i] << ", rel diff " << hrel_diff[i]
                          << std::endl;
            }
            if(habs_diff[i] > tolerance + tolerance * abs(hb_float[i]))
            {
                failed = true;
            }
            if(habs_diff[i] > max_abs_diff)
            {
                max_abs_diff = habs_diff[i];
            }
            if(hrel_diff[i] > max_rel_diff)
            {
                max_rel_diff = hrel_diff[i];
            }
        }
    }
    validate_data_t return_data;
    return_data.pass            = !failed;
    return_data.numElements     = count;
    return_data.maxAbsoluteDiff = max_abs_diff;
    return_data.maxRelativeDiff = max_rel_diff;
    return_data.tolerance       = tolerance;

    return return_data;
}

#endif // DLRM_TEST_COMMON_H
