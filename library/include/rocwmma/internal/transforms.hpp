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
    template <uint32_t BlockDim, uint32_t VW>
    struct AosToSoa;

    template <uint32_t BlockDim, uint32_t VW>
    struct SoaToAos;

    ///
    /// AOS -> SOA : Transform from inline VW to ortho VW
    ///

    // BlockDim = 16
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_16xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 32
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_32xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 64
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_64xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 128
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_128xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 256
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto aos_soa_256xk_b32(VecT<DataT, 2> const& v);

    //
    /// SOA -> AOS : Transform from ortho VW to inline VW
    ///

    // BlockDim = 16
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_16xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 32
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_32xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 64
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_64xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 128
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_128xk_b32(VecT<DataT, 2> const& v);

    // BlockDim = 256
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 8> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 4> const& v);
    template <typename DataT>
    ROCWMMA_DEVICE static inline auto soa_aos_256xk_b32(VecT<DataT, 2> const& v);

} // namespace rocwmma

#include "transforms_impl.hpp"

#endif // ROCWMMA_TRANSFORMS_HPP
