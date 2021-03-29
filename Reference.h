#ifndef WMMA_REFERENCE_H
#define WMMA_REFERENCE_H

#include "Types.h"
#include <type_traits>

template <typename InputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
void gemm_CPU(uint32_t        m,
              uint32_t        n,
              uint32_t        k,
              InputT const*   a,
              InputT const*   b,
              ComputeT const* c,
              ComputeT*       d,
              ComputeT        alpha,
              ComputeT        beta)
{
    int lda = std::is_same<LayoutA, row_major>::value ? k : m;
    int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
    int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
    int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;
    auto cIndex = std::is_same<LayoutC, row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, row_major>::value ? rowMjr : colMjr;

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
            d[dIndex(i, j, ldd)] = alpha * accum + beta * c[cIndex(i, j, ldc)];
        }
    }
}

#endif // WMMA_REFERENCE_H
