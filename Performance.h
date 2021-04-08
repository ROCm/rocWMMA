#ifndef WMMA_PERFORMANCE_H
#define WMMA_PERFORMANCE_H

#include "Types.h"

template <typename DataT>
struct MfmaPerfTraits;

template <>
struct MfmaPerfTraits<float16_t>
{
    enum : uint32_t
    {
        Multiplier = 16
    };
};

template <>
struct MfmaPerfTraits<__half> : public MfmaPerfTraits<float16_t>
{
};

template <>
struct MfmaPerfTraits<float32_t>
{
    enum : uint32_t
    {
        Multiplier = 4
    };
};

template <typename DataT>
struct VALUPerfTraits;

template <>
struct VALUPerfTraits<float16_t>
{
    enum : uint32_t
    {
        Multiplier = 4
    };
};

template <>
struct VALUPerfTraits<__half> : public VALUPerfTraits<float16_t>
{
};

template <>
struct VALUPerfTraits<float32_t>
{
    enum : uint32_t
    {
        Multiplier = 2
    };
};

class Mi100;
class Vega20;

template <typename GfxArch>
struct HardwareTraits;

template <>
struct HardwareTraits<Mi100>
{
    enum : uint32_t
    {
        CuCount = 120,
    };
};

inline double calculateAccumGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    constexpr double flopsPerAccum = 2.0;
    return flopsPerAccum * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)
           / 1000000000.0;
}

inline double calculateBlendGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    constexpr double flopsPerBlend = 3.0;
    return flopsPerBlend * static_cast<double>(m) * static_cast<double>(n) / 1000000000.0;
}

inline double calculateTotalGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    return calculateAccumGFlops(m, n, k) + calculateBlendGFlops(m, n, k);
}

inline double calculateGFlopsPerSec(uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs)
{
    return calculateTotalGFlops(m, n, k) / elapsedTimeMs * 1000.0;
}

template <typename InputT,
          typename ComputeT,
          typename GfxArch,
          template <typename> class AccumPerfTraits = MfmaPerfTraits,
          template <typename> class BlendPerfTraits = VALUPerfTraits>
inline double calculatePeakGFlopsPerSec(uint32_t M, uint32_t N, uint32_t K, uint32_t freqMHz)
{
    auto accumGFlops = calculateAccumGFlops(M, N, K);
    auto blendGFlops = calculateBlendGFlops(M, N, K);
    auto totalGFlops = accumGFlops + blendGFlops;

    auto accumMultiplier = static_cast<double>(AccumPerfTraits<InputT>::Multiplier);
    auto blendMultiplier = static_cast<double>(BlendPerfTraits<ComputeT>::Multiplier);

    auto basePeakGFlops
        = static_cast<double>(64.0 * HardwareTraits<GfxArch>::CuCount * freqMHz) / 1000.0;

    return (accumGFlops / totalGFlops * accumMultiplier * basePeakGFlops)
           + // Portion of peak flops in accumulation
           (blendGFlops / totalGFlops * blendMultiplier
            * basePeakGFlops); // Portion of peak flops in blending
}

#endif // WMMA_PERFORMANCE_H
