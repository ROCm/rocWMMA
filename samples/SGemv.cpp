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

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <WMMA/WMMA.h>

#include "Common.h"

using rocwmma::float16_t;
using rocwmma::float32_t;
using rocwmma::float64_t;

// Host gemv validation
__host__ void gemv_cpu_h(uint32_t         m,
                         uint32_t         n,
                         uint32_t         k,
                         float16_t const* a,
                         float16_t const* b,
                         float32_t*       c,
                         float32_t        alpha,
                         float32_t        beta)
{
    uint32_t lda = m;

    for(int i = 0; i < m; ++i)
    {
        float32_t accum = 0.0f;
        for(int h = 0; h < k; ++h)
        {
            accum += static_cast<float32_t>(a[i * lda + h]) * static_cast<float32_t>(b[h]);
        }
        c[i] = alpha * accum + beta * c[i];
    }
}

// Matrix data print
template <typename DataT>
__host__ static inline void print(DataT* mat, uint32_t m, uint32_t n)
{
    auto ld = n;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            std::cout << static_cast<DataT>(mat[i * ld + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
{
    uint32_t ld = n;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            // Ascending order for each neighboring element.
            // Alternate sign for even / odd
            auto value      = (i * ld + j) % 17;
            mat[i * ld + j] = (value % 2) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
        }
    }
}

// Element-wise comparison
__host__ void
    compareEqual(float32_t const* a, float32_t const* b, int m, int n, double tolerance = 10.0)
{
    bool retval;
    int  lda = n;
    int  ldb = lda;

    double max_relative_error = 0.0;

    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            auto valA           = a[i * lda + j];
            auto valB           = b[i * ldb + j];
            auto relative_error = fabs(valA - valB) / (fabs(valA) + fabs(valB) + 1.0);

            if(relative_error > max_relative_error || relative_error != relative_error)
            {
                max_relative_error = relative_error;
            }
        }
    }

    auto eps = std::numeric_limits<float32_t>::epsilon();
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

// Supports WMMA_M/N square sizes of
// : 16 x 16
// : 32 x 32
const int WMMA_M = 16;
const int WMMA_N = 16;

// Supports WMMA_K sizes as
// : multiples of 16.
const int WMMA_K = 16;

// AMDGCN default wave size
const int WAVE_SIZE = 64;

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X = 16 * WAVE_SIZE;
const int T_BLOCK_Y = 1;

// The following device kernel is a naive implementation
// of blocked GEMV. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the m x k x 1 GEMV, generalized as:
// y = alpha * (A) * x + beta * y
//
// In this simplified example, we assume:
//  A - Matrix of size m * k (row-major)
//  x - Vector of size k * 1 (col-major)
//  y - accumulator of size m * 1 (row-major)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Note: This is a simplified implementation to demonstrate API usage in
// context of wave-level GEMV computation, and is not optimized.
__global__ void gemv_wmma_d(uint32_t         m,
                            uint32_t         n,
                            uint32_t         k,
                            float16_t const* a,
                            float16_t const* b,
                            float32_t*       c,
                            float32_t        alpha,
                            float32_t        beta)
{
    uint32_t lda = m;
    uint32_t ldb = k;
    uint32_t ldc = m;

    // Global warp id, warpN is 0.
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;

    // Declare the fragments
    rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float16_t, rocwmma::row_major>
        a_frag;
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float16_t, rocwmma::col_major>
                                                                               b_frag;
    rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float32_t> acc_frag;
    rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float32_t> c_frag;

    rocwmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for(int i = 0; i < k; i += WMMA_K)
    {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;

        // Bounds checking
        if(aRow < m)
        {
            // Load the inputs
            rocwmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            rocwmma::load_matrix_sync(b_frag, b + bRow, ldb);

            // Perform the matrix vector multiplication
            rocwmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cRow = warpM * WMMA_M;

    if(cRow < m)
    {
        rocwmma::load_matrix_sync(c_frag, c + cRow, ldc, rocwmma::mem_col_major);

        for(int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        rocwmma::store_matrix_sync(c + cRow, c_frag, ldc, rocwmma::mem_col_major);
    }
}

__host__ void gemv_test(uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    // Bounds check
    if(m % WMMA_M || n % WMMA_N || k % WMMA_K)
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    std::cout << "Initializing host data..." << std::endl;
    // Initialize input matrices
    std::vector<float16_t> matrixA(m * k); // matrix
    std::vector<float16_t> matrixB(k * 1); // vector
    std::vector<float32_t> matrixC(m * 1, 1.0); //accum

    fill(matrixA.data(), m, k);
    fill(matrixB.data(), k, 1);

    std::cout << "Initializing device data..." << std::endl;
    // Allocate and copy device memory
    float16_t* d_a;
    float16_t* d_b;
    float32_t* d_c;

    const size_t bytesA = matrixA.size() * sizeof(float16_t);
    const size_t bytesB = matrixB.size() * sizeof(float16_t);
    const size_t bytesC = matrixC.size() * sizeof(float32_t);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, WMMA_M * T_BLOCK_X / WAVE_SIZE),
                         rocwmma::ceilDiv(n, WMMA_N * T_BLOCK_Y));

    std::cout << "Launching gemv kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    hipExtLaunchKernelGGL(gemv_wmma_d,
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          alpha,
                          beta);

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, " << std::endl;

    std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", " << n << ", " << k
              << std::endl;

    std::cout << "Validating result with reference..." << std::endl;
    // Bring kernel result back to host
    std::vector<float32_t> matrixC_device(m * 1, std::numeric_limits<float32_t>::signaling_NaN());
    CHECK_HIP_ERROR(hipMemcpy(matrixC_device.data(), d_c, bytesC, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<float32_t> matrixC_host(matrixC);
    gemv_cpu_h(m, n, k, matrixA.data(), matrixB.data(), matrixC_host.data(), alpha, beta);

    compareEqual(matrixC_host.data(), matrixC_device.data(), m, 1u);

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    const uint32_t m = 256;
    const uint32_t k = 256;
    const uint32_t n = T_BLOCK_Y * WMMA_N;

    gemv_test(m, n, k, 2.1f, 2.1f);
    return 0;
}
