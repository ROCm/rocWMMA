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
#ifndef WMMA_REFERENCE_H
#define WMMA_REFERENCE_H

#include "Types.h"
#include <type_traits>

namespace rocwmma
{

    template <typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void gemm_CPU(uint32_t       m,
                  uint32_t       n,
                  uint32_t       k,
                  InputT const*  a,
                  InputT const*  b,
                  OutputT const* c,
                  OutputT*       d,
                  ComputeT       alpha,
                  ComputeT       beta)
    {
        int lda = std::is_same<LayoutA, rocwmma::row_major>::value ? k : m;
        int ldb = std::is_same<LayoutB, rocwmma::row_major>::value ? n : k;
        int ldc = std::is_same<LayoutC, rocwmma::row_major>::value ? n : m;
        int ldd = std::is_same<LayoutD, rocwmma::row_major>::value ? n : m;

        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto aIndex = std::is_same<LayoutA, rocwmma::row_major>::value ? rowMjr : colMjr;
        auto bIndex = std::is_same<LayoutB, rocwmma::row_major>::value ? rowMjr : colMjr;
        auto cIndex = std::is_same<LayoutC, rocwmma::row_major>::value ? rowMjr : colMjr;
        auto dIndex = std::is_same<LayoutD, rocwmma::row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                ComputeT accum = static_cast<ComputeT>(0);
                for(int h = 0; h < k; ++h)
                {
                    accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                             * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
                }
                d[dIndex(i, j, ldd)] = static_cast<OutputT>(
                    alpha * accum + beta * static_cast<ComputeT>(c[cIndex(i, j, ldc)]));
            }
        }
    }

} // namespace rocwmma

template <typename DataT>
void dlrm_fwd_CPU(DataT const* input, DataT* output, uint32_t m, uint32_t k, uint32_t batchSize)
{
    auto batchOffset       = m * k;
    uint outputBatchOffset = ((m * (m - 1)) / 2) + k;
#pragma omp parallel for
    for(int b = 0; b < batchSize; b++)
    {
        uint outputIdx = b * outputBatchOffset;

        // Copy MLP to output
        for(int i = 0; i < k; i++)
        {

            output[outputIdx] = input[b * batchOffset + i];
            outputIdx++;
        }
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < m; j++)
            {
                float accum = static_cast<float>(0);
                for(int h = 0; h < k; h++)
                {
                    accum += static_cast<float>(input[b * batchOffset + i * k + h])
                             * static_cast<float>(input[b * batchOffset + j * k + h]);
                }

                if(j < i)
                {
                    output[outputIdx] = static_cast<DataT>(accum);
                    outputIdx++;
                }
            }
        }
    }
}

template <typename DataT>
void dlrm_bwd_CPU(DataT const* input,
                  DataT const* upstreamGrad,
                  DataT*       bottomMlpGrad,
                  DataT*       output,
                  uint32_t     m,
                  uint32_t     k,
                  uint32_t     batchSize)
{
    auto batchOffset = m * k;
    auto accOffset   = m * m;
    auto trilSize    = ((m * (m - 1)) / 2) + k;
    auto acc         = new DataT[batchSize * m * m];

#pragma omp parallel for
    for(int b = 0; b < batchSize; b++)
    {
        // Copy bottom MLP grad
        for(int j = 0; j < k; j++)
        {
            bottomMlpGrad[b * k + j] = upstreamGrad[b * trilSize + j];
        }

        // Remake tril
        uint32_t upstreamIdx = b * trilSize + k;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                if(i == j)
                {
                    acc[b * accOffset + i * m + j] = 0;
                }
                else
                {
                    acc[b * accOffset + i * m + j] = upstreamGrad[upstreamIdx];
                    acc[b * accOffset + j * m + i] = upstreamGrad[upstreamIdx];
                    upstreamIdx++;
                }
            }
        }

        // Perform reverse bmm
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < k; j++)
            {
                float accum = 0.0f;
                for(int h = 0; h < m; h++)
                {
                    accum += static_cast<float>(acc[b * accOffset + i * m + h])
                             * static_cast<float>(input[b * batchOffset + h * k + j]);
                }
                output[b * batchOffset + i * k + j] = static_cast<DataT>(accum);
            }
        }
    }
    delete[] acc;
}
#endif // WMMA_REFERENCE_H
