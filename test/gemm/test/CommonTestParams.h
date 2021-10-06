#ifndef WMMA_GEMM_COMMON_TEST_PARAMS_H
#define WMMA_GEMM_COMMON_TEST_PARAMS_H

#include <tuple>
#include <vector>

#include "Common.hpp"
#include "Types.h"

struct CommonTestParams
{
    // Testing types as Input/Output/Compute (IOC)
    using TestTypesIOC = std::tuple<
        // Non-native bfloat16_t
        std::tuple<bfloat16_t, bfloat16_t, bfloat16_t>,
        std::tuple<bfloat16_t, bfloat16_t, float32_t>,
        std::tuple<bfloat16_t, float32_t, float32_t>,

        // Native fp16
        std::tuple<float16_t, float16_t, float16_t>,
        std::tuple<float16_t, float16_t, float32_t>,
        std::tuple<float16_t, float32_t, float32_t>,

        // Native fp32
        std::tuple<float32_t, float32_t, float32_t>,

        // Non-native hfloat16_t (i.e. __half)
        std::tuple<hfloat16_t, hfloat16_t, hfloat16_t>,
        std::tuple<hfloat16_t, hfloat16_t, float32_t>,
        std::tuple<hfloat16_t, float32_t, float32_t>,

        // Native int8
        std::tuple<int8_t, int32_t, int32_t>,
        std::tuple<int8_t, int8_t, int32_t>>;

    using TestTypeDouble = std::tuple<
        // Native double
        std::tuple<float64_t, float64_t, float64_t>>;

    using TestBlockSizes16x16 = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                           std::tuple<I<16>, I<16>, I<32>>,
                                           std::tuple<I<16>, I<16>, I<64>>,
                                           std::tuple<I<16>, I<16>, I<128>>,
                                           std::tuple<I<16>, I<16>, I<256>>>;

    using TestBlockSizes32x32 = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                           std::tuple<I<32>, I<32>, I<16>>,
                                           std::tuple<I<32>, I<32>, I<32>>,
                                           std::tuple<I<32>, I<32>, I<64>>,
                                           std::tuple<I<32>, I<32>, I<128>>>;

    // Supported layout types
    using TestLayoutTypes = std::tuple<row_major, col_major>;

    using ThreadBlockT = std::pair<int64_t, int64_t>;
    using ProblemSizeT = std::tuple<int64_t, int64_t, int64_t>;
    using AlphaT       = float64_t;
    using BetaT        = float64_t;

    static inline std::vector<ThreadBlockT> threadBlocks()
    {
        return {{64, 1}, {64, 2}, {64, 4}, {128, 1}, {128, 2}, {256, 1}};
    }

    static inline std::vector<ProblemSizeT> problemSizes()
    {
        return {{64, 64, 1024},
                {32, 64, 1024},
                {64, 32, 1024},
                {256, 256, 1024},
                {2048, 64, 1024},
                {64, 2048, 1024},
                {1024, 1024, 1024}
#ifndef WMMA_VALIDATE_TESTS
                ,
                {2048, 2048, 2048},
                {2560, 2560, 2560},
                {3072, 3072, 3072},
                {3584, 3584, 3584},
                {4096, 4096, 4096},
                {5120, 5120, 5120},
                {6144, 6144, 6144},
                {7168, 7168, 7168},
                {8192, 8192, 8192}
#endif // WMMA_VALIDATE_TESTS
        };
    }

    static inline std::vector<AlphaT> alphas()
    {
        return {2.0};
    }

    static inline std::vector<BetaT> betas()
    {
        return {2.0};
    }
};

#endif // WMMA_GEMM_COMMON_TEST_PARAMS_H
