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
#ifndef ROCWMMA_TRANSFORMS_HPP
#define ROCWMMA_TRANSFORMS_HPP

#include "vector.hpp"

namespace rocwmma
{
    // AOS -> SOA
    // Transform FROM mem friendly layout TO MFMA friendly layout
    template <typename DataT>
    __device__ void aos_soa_16x32_vw8_b32_opt(VecT<DataT, 8>& v);
    template <typename DataT>
    __device__ void aos_soa_16x16_vw4_b32_opt(VecT<DataT, 4>& v);
    template <typename DataT>
    __device__ void aos_soa_16x8_vw2_b32_opt(VecT<DataT, 2>& v);

    template <typename DataT>
    __device__ void aos_soa_32x4_vw2_b32_opt(VecT<DataT, 2>& v);

    // SOA -> AOS
    // Transform FROM MFMA friendly layout to mem friendly layout
    template <typename DataT>
    __device__ void soa_aos_16x16_vw4_b32_opt(VecT<DataT, 4>& v);
    template <typename DataT>
    __device__ void soa_aos_16x8_vw2_b32_opt(VecT<DataT, 2>& v);

} // namespace rocwmma

#include "transforms_impl.hpp"

#endif // ROCWMMA_TRANSFORMS_HPP
