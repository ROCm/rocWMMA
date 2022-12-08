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

std::string source = R"(

#include <rocwmma/rocwmma.hpp>

extern "C"

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
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
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
