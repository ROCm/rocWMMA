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

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "Types.h"
#include "Utils.h"
#include "WMMA.h"

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

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

struct buff_t
{
    void*       h;
    void*       d;
    size_t      bytes;
    std::string file;
};

// bool read_buff(struct buff_t* buff, bool to_device = true, bool verbose = false)
// {
//     const char* filename = buff->file.c_str();
//     void**      hbuff    = &buff->h;
//     void**      dbuff    = &buff->d;

//     int fd = open(filename, O_RDONLY);
//     if(fd < 0)
//     {
//         std::cout << "Invalid file " << filename << std::endl;
//         return false;
//     }
//     struct stat stats;

//     fstat(fd, &stats);
//     buff->bytes = stats.st_size;

//     std::cout << "Read " << filename << " bytes " << buff->bytes << std::endl;

//     *hbuff = malloc(buff->bytes);
//     hipMalloc(dbuff, buff->bytes);

//     pread(fd, *hbuff, buff->bytes, 0);
//     if(to_device)
//     {
//         hipMemcpy(*dbuff, *hbuff, buff->bytes, hipMemcpyDefault);
//     }

//     if(verbose)
//     {
//         for(int i = 0; i < buff->bytes / sizeof(float); i++)
//         {
//             std::cout << ((float*)*hbuff)[i] << ", ";
//             if(i != 0 && i % 100 == 0)
//                 std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     close(fd);
//     return true;
// }

// bool write_buff(const char* filename, void* buff, size_t bytes)
// {
//     int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC);
//     if(fd < 0)
//     {
//         std::cout << "Invalid file " << filename << std::endl;
//         return false;
//     }

//     pwrite(fd, buff, bytes, 0);

//     close(fd);
//     return true;
// }

__device__ inline bool is_same(half a, half b)
{
    return __heq(a, b);
}

__device__ inline bool is_same(float a, float b)
{
    return a == b;
}

template <typename T, uint THREADBLOCK_SIZE>
__global__ __launch_bounds__(THREADBLOCK_SIZE) void allclose_kernel(
    T* a, T* b, size_t num_elm, float* abs_diff, float* rel_diff, float* a_float, float* b_float)
{
    int    tid      = threadIdx.x;
    int    nthreads = blockDim.x;
    size_t start    = (num_elm * tid) / nthreads;
    size_t end      = (num_elm * (tid + 1)) / nthreads;
    for(size_t i = start; i < end; i++)
    {
        if(!is_same(a[i], b[i]))
        {
            float a_    = (float)a[i];
            float b_    = (float)b[i];
            a_float[i]  = a_;
            b_float[i]  = b_;
            abs_diff[i] = fabs(a_ - b_);
            if(a_ != 0.0f)
            {
                rel_diff[i] = abs_diff[i] / fabs(a_);
            }
            else
            {
                rel_diff[i] = 0.0f;
            }
        }
        else
        {
            abs_diff[i] = 0.0f;
            rel_diff[i] = 0.0f;
        }
    }
}

template <typename T>
bool allclose(void* a, void* b, size_t bytes, bool verbose = false)
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

    if(failed)
    {
        std::cout << "FAIL: ";
    }
    else
    {
        std::cout << "PASS: ";
    }

    if(count == 0)
    {
        std::cout << "Identical" << std::endl;
    }
    else
    {
        std::cout << "Not identical" << std::endl
                  << ">>> Num non-identical elements: " << count << std::endl
                  << ">>> Max absolute diff: " << max_abs_diff << std::endl
                  << ">>> Max relative diff: " << max_rel_diff << std::endl
                  << ">>> Tolerance: " << tolerance << std::endl;
    }
    return !failed;
}

__device__ static inline void syncwarp()
{
#ifdef __HIP_PLATFORM_HCC__
    __builtin_amdgcn_wave_barrier();
#else
    __syncwarp();
#endif
}

// enum CmdOptionType
// {
//     INT_OPT,
//     LONG_OPT,
//     BOOL_OPT,
//     HELP_OPT,
// };

// struct CmdOption
// {
//     void*         val;
//     CmdOptionType type;
//     const char*   name;
//     int           has_arg;
//     const char*   description;
// };

// void print_help(const char* bin, std::map<int, CmdOption>& options)
// {
//     printf("Usage: %s [OPTIONS]\n", bin);
//     for(auto m : options)
//     {
//         auto k = m.first;
//         auto v = m.second;

//         std::string name = v.name;
//         std::string arg = "", desc = "";
//         int         space_width = 30;

//         if(v.has_arg == required_argument)
//         {
//             arg = " " + name;
//             transform(
//                 arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return toupper(c); });
//             space_width -= arg.size();
//         }

//         name = "--" + name;
//         if((k >= 'a' && k <= 'z') || (k >= 'A' && k <= 'Z'))
//         {
//             std::string short_opt(1, (char)k);
//             name = "-" + short_opt + " | " + name;
//         }

//         space_width -= name.size();
//         if(v.description == nullptr)
//         {
//             space_width = 0;
//         }
//         else
//         {
//             desc = v.description;
//         }

//         printf("    %s%s%-*s%s\n", name.c_str(), arg.c_str(), space_width, " ", desc.c_str());
//     }
//     exit(0);
// }

// void get_options(int argc, char** argv, std::map<int, CmdOption>& options)
// {
//     int c;
//     int option_index = 0;

//     std::vector<struct option> long_options;
//     char**                     help_description;
//     std::string                short_options = "";

//     // Construct options
//     int i = 0;
//     for(auto m : options)
//     {
//         auto        k    = m.first;
//         auto        v    = m.second;
//         const char* name = v.name;
//         long_options.push_back({name, v.has_arg, 0, k});

//         if((k >= 'a' && k <= 'z') || (k >= 'A' && k <= 'Z'))
//         {
//             std::string short_opt(1, (char)k);
//             short_options += short_opt;
//             if(v.has_arg != no_argument)
//             {
//                 short_options += ":";
//             }
//         }
//         i++;
//     }

//     while(true)
//     {
//         c = getopt_long(argc, argv, short_options.c_str(), long_options.data(), &option_index);
//         if(c == -1)
//             break;

//         auto item = options.find(c);
//         if(item != options.end())
//         {
//             auto opt = item->second;
//             switch(opt.type)
//             {
//             case HELP_OPT:
//             {
//                 print_help(argv[0], options);
//                 break;
//             }
//             case INT_OPT:
//             {
//                 int val = atoi(optarg);
//                 memcpy(opt.val, &val, sizeof(int));
//                 printf("Setting %s to %d\n", opt.name, val);
//                 break;
//             }
//             case LONG_OPT:
//             {
//                 int64_t val = atol(optarg);
//                 memcpy(opt.val, &val, sizeof(int64_t));
//                 printf("Setting %s to %ld\n", opt.name, val);
//                 break;
//             }
//             case BOOL_OPT:
//             {
//                 bool val = true;
//                 memcpy(opt.val, &val, sizeof(bool));
//                 printf("Enabling %s\n", opt.name);
//                 break;
//             }
//             }
//         }
//         else
//         {
//             printf("Unknown option %d\n", c);
//         }
//     }
// }

#endif // DLRM_TEST_COMMON_H
