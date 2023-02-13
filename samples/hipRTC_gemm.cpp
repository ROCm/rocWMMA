/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <rocwmma/rocwmma.hpp>

#include "common.hpp"

using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;

// Host GEMM validation
__host__ void gemm_cpu_h(uint32_t         m,
                         uint32_t         n,
                         uint32_t         k,
                         float16_t const* a,
                         float16_t const* b,
                         float32_t const* c,
                         float32_t*       d,
                         uint32_t         lda,
                         uint32_t         ldb,
                         uint32_t         ldc,
                         uint32_t         ldd,
                         float32_t        alpha,
                         float32_t        beta)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            float32_t accum = 0.0f;
            for(int h = 0; h < k; ++h)
            {
                accum += static_cast<float32_t>(a[i * lda + h])
                         * static_cast<float32_t>(b[j * ldb + h]);
            }
            d[i * ldd + j] = alpha * accum + beta * c[i * ldc + j];
        }
    }
}

// Supports ROCWMMA_M/N square sizes of
// : 16 x 16
// : 32 x 32
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;

// Supports ROCWMMA_K sizes as
// : multiples of 16.
const int ROCWMMA_K = 16;

// Device warp size
const int WAVE_SIZE = getWarpSize();

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

std::string source = R"(

#include <rocwmma/rocwmma.hpp>

using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;

constexpr int ROCWMMA_M = 16;
constexpr int ROCWMMA_N = 16;
constexpr int ROCWMMA_K = 16;

// The following device kernel is a naive implementation
// of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the M x N x K GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this simplified example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level GEMM computation, and is not optimized.
extern "C"
__global__ void gemm_rocwmma_d(uint32_t         m,
                               uint32_t         n,
                               uint32_t         k,
                               float16_t const* a,
                               float16_t const* b,
                               float32_t const* c,
                               float32_t*       d,
                               uint32_t         lda,
                               uint32_t         ldb,
                               uint32_t         ldc,
                               uint32_t         ldd,
                               float32_t        alpha,
                               float32_t        beta)
{
    // Create frags
    auto fragA = rocwmma::fragment<rocwmma::matrix_a,
                                   ROCWMMA_M,
                                   ROCWMMA_N,
                                   ROCWMMA_K,
                                   float16_t,
                                   rocwmma::row_major>();
    auto fragB = rocwmma::fragment<rocwmma::matrix_b,
                                   ROCWMMA_M,
                                   ROCWMMA_N,
                                   ROCWMMA_K,
                                   float16_t,
                                   rocwmma::col_major>();
    auto fragC
        = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();
    auto fragAcc
        = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

    rocwmma::fill_fragment(fragAcc, 0.0f);

    // Tile using a 2D grid
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / rocwmma::Constants::AMDGCN_WAVE_SIZE;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    // Target C block
    auto cRow = majorWarp * ROCWMMA_M;
    auto cCol = minorWarp * ROCWMMA_N;

    // Bounds check
    if(cRow < m && cCol < n)
    {
        // fragAcc = A x B
        for(int i = 0; i < k; i += ROCWMMA_K)
        {
            // Load the inputs
            rocwmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
            rocwmma::load_matrix_sync(fragB, b + (i + cCol * ldb), ldb);

            // Matrix multiply - accumulate using MFMA units
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Fetch C matrix
        rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

        // D = alpha * A x B + beta * C
        for(int i = 0; i < fragC.num_elements; ++i)
        {
            fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
        }

        // Store to D
        rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);
    }
}
)";

char const* src = source.c_str();

int main()
{
    /// Determine the rocm path to use for build
    // 1. Environment variable
    // 2. Default path
    std::string rocm_path
        = (std::getenv("ROCM_PATH") == nullptr) ? "/opt/rocm" : std::getenv("ROCM_PATH");
    std::string rocWMMAIncludePath = std::string("-I") + rocm_path + std::string("/include");

    // gemm parameters
    uint32_t  m     = 256;
    uint32_t  n     = 256;
    uint32_t  k     = 256;
    float32_t alpha = 2.1f;
    float32_t beta  = 2.1f;

    hiprtcProgram prog;
    CHECK_HIPRTC_ERROR(hiprtcCreateProgram(&prog, src, nullptr, 0, nullptr, nullptr));
    hiprtcResult result;
    hiprtcResult logResult;
    const char*  opts[] = {"-D__HIP_PLATFORM_AMD__", rocWMMAIncludePath.c_str()};

    result = hiprtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts);
    if(result != HIPRTC_SUCCESS)
    {
        std::cout << "HipRTC compile failed." << std::endl;
        std::cout << result << std::endl;
        std::string s_error = hiprtcGetErrorString(result);
        std::cout << s_error << std::endl;
        std::size_t log_size;
        CHECK_HIPRTC_ERROR(hiprtcGetProgramLogSize(prog, &log_size));
        std::cout << "Log Size: " << log_size << std::endl;
        std::string log;

        log.reserve(log_size);
        CHECK_HIPRTC_ERROR(hiprtcGetProgramLog(prog, &log[0]));

        std::cout << log.c_str() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t code_size;
    CHECK_HIPRTC_ERROR(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);

    CHECK_HIPRTC_ERROR(hiprtcGetCode(prog, code.data()));

    hipModule_t   module;
    hipFunction_t func;
    CHECK_HIP_ERROR(hipModuleLoadData(&module, code.data()));
    CHECK_HIP_ERROR(hipModuleGetFunction(&func, module, "gemm_rocwmma_d"));

    // Bounds check
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
       || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
    {
        std::cout << "Unsupported size!\n";
        return 0;
    }

    uint32_t lda = k;
    uint32_t ldb = k;
    uint32_t ldc = n;
    uint32_t ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<float16_t> matrixA(m * k);
    std::vector<float16_t> matrixB(k * n);
    std::vector<float32_t> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

    fill(matrixA.data(), m, k);
    fill(matrixB.data(), k, n);
    fill(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    hipDeviceptr_t d_a, d_b, d_c, d_d;

    const size_t bytesA = matrixA.size() * sizeof(float16_t);
    const size_t bytesB = matrixB.size() * sizeof(float16_t);
    const size_t bytesC = matrixC.size() * sizeof(float32_t);
    const size_t bytesD = matrixD.size() * sizeof(float32_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                        rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));

    struct
    {
        uint32_t       _m;
        uint32_t       _n;
        uint32_t       _k;
        hipDeviceptr_t _d_a;
        hipDeviceptr_t _d_b;
        hipDeviceptr_t _d_c;
        hipDeviceptr_t _d_d;
        uint32_t       _lda;
        uint32_t       _ldb;
        uint32_t       _ldc;
        uint32_t       _ldd;
        float32_t      _alpha;
        float32_t      _beta;
    } args{m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta};

    std::size_t args_size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size,
                      HIP_LAUNCH_PARAM_END};

    std::cout << "Launching GEMM kernel..." << std::endl;

    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    CHECK_HIP_ERROR(hipModuleLaunchKernel(func,
                                          gridDim.x,
                                          gridDim.y,
                                          gridDim.z,
                                          blockDim.x,
                                          blockDim.y,
                                          blockDim.z,
                                          0,
                                          nullptr,
                                          nullptr,
                                          (void**)&config));

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // GEMM flops converge to 2*mnk
    auto gFlops       = 2.0 * static_cast<double>(m * n * k) * 1.0e-9;
    auto tFlopsPerSec = gFlops / static_cast<double>(elapsedTimeMs);

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << ROCWMMA_M << ", " << ROCWMMA_N << ", " << ROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float32_t> matrixD_ref(m * n, std::numeric_limits<float32_t>::signaling_NaN());
    gemm_cpu_h(m,
               n,
               k,
               matrixA.data(),
               matrixB.data(),
               matrixC.data(),
               matrixD_ref.data(),
               lda,
               ldb,
               ldc,
               ldd,
               alpha,
               beta);

    compareEqual<float32_t>(matrixD.data(), matrixD_ref.data(), m * n);

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;

    CHECK_HIP_ERROR(hipModuleUnload(module));
    CHECK_HIPRTC_ERROR(hiprtcDestroyProgram(&prog));

    return 0;
}
