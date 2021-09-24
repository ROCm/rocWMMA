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
#ifndef WMMA_PERFORMANCE_H
#define WMMA_PERFORMANCE_H

#include "Types.h"

// Architectures
class MI100;
class MI200;
class Vega20;

template <typename GfxArch, typename DataT>
struct MfmaPerfTraits;

// MI-100
template <>
struct MfmaPerfTraits<MI100, int8_t>
{
    enum : uint32_t
    {
        Multiplier = 1024
    };
};

template <>
struct MfmaPerfTraits<MI100, bfloat16_t>
{
    enum : uint32_t
    {
        Multiplier = 512
    };
};

template <>
struct MfmaPerfTraits<MI100, float16_t>
{
    enum : uint32_t
    {
        Multiplier = 1024
    };
};

template <>
struct MfmaPerfTraits<MI100, float32_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct MfmaPerfTraits<MI100, float64_t>
{
    enum : uint32_t
    {
        Multiplier = 0
    };
};

// MI-200
template <>
struct MfmaPerfTraits<MI200, int8_t>
{
    enum : uint32_t
    {
        Multiplier = 1024
    };
};

template <>
struct MfmaPerfTraits<MI200, bfloat16_t>
{
    enum : uint32_t
    {
        Multiplier = 512
    };
};

template <>
struct MfmaPerfTraits<MI200, float16_t>
{
    enum : uint32_t
    {
        Multiplier = 1024
    };
};

template <>
struct MfmaPerfTraits<MI200, float32_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct MfmaPerfTraits<MI200, float64_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <typename GfxArch>
struct MfmaPerfTraits<GfxArch, hfloat16_t> : public MfmaPerfTraits<GfxArch, float16_t>
{
};

template <typename GfxArch, typename DataT>
struct VALUPerfTraits;

// MI-100
template <>
struct VALUPerfTraits<MI100, int8_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct VALUPerfTraits<MI100, bfloat16_t>
{
    enum : uint32_t
    {
        Multiplier = 128
    };
};

template <>
struct VALUPerfTraits<MI100, float16_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct VALUPerfTraits<MI100, float32_t>
{
    enum : uint32_t
    {
        Multiplier = 128
    };
};

template <>
struct VALUPerfTraits<MI100, float64_t>
{
    enum : uint32_t
    {
        Multiplier = 64
    };
};

// MI200
template <>
struct VALUPerfTraits<MI200, int8_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct VALUPerfTraits<MI200, bfloat16_t>
{
    enum : uint32_t
    {
        Multiplier = 128
    };
};

template <>
struct VALUPerfTraits<MI200, float16_t>
{
    enum : uint32_t
    {
        Multiplier = 256
    };
};

template <>
struct VALUPerfTraits<MI200, float32_t>
{
    enum : uint32_t
    {
        Multiplier = 128
    };
};

template <>
struct VALUPerfTraits<MI200, float64_t>
{
    enum : uint32_t
    {
        Multiplier = 128
    };
};

template <typename GfxArch>
struct VALUPerfTraits<GfxArch, hfloat16_t> : public VALUPerfTraits<GfxArch, float16_t>
{
};

template <typename GfxArch>
struct HardwareTraits;

template <>
struct HardwareTraits<MI100>
{
    enum : uint32_t
    {
        CuCount = 120,
    };
};

template <>
struct HardwareTraits<MI200>
{
    enum : uint32_t
    {
        CuCount = 110,
    };
};

inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)
           / 1000000000.0;
}

inline double calculateGFlopsPerSec(uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs)
{
    return calculateGFlops(m, n, k) / elapsedTimeMs * 1000.0;
}

template <typename InputT,
          typename GfxArch,
          template <typename, typename> class PerfTraits = MfmaPerfTraits>
inline double calculatePeakGFlopsPerSec(uint32_t freqMHz)
{
    return static_cast<double>(PerfTraits<GfxArch, InputT>::Multiplier)
           * static_cast<double>(HardwareTraits<GfxArch>::CuCount) * static_cast<double>(freqMHz)
           / 1000.0;
}

#endif // WMMA_PERFORMANCE_H
