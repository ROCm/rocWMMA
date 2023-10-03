/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_SAMPLES_COMMON_HPP
#define ROCWMMA_SAMPLES_COMMON_HPP

#include <iostream>
#include <mutex>

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPRTC_ERROR
#define CHECK_HIPRTC_ERROR(status)                   \
    if(status != HIPRTC_SUCCESS)                     \
    {                                                \
        fprintf(stderr,                              \
                "hipRTC error: '%s'(%d) at %s:%d\n", \
                hiprtcGetErrorString(status),        \
                status,                              \
                __FILE__,                            \
                __LINE__);                           \
        exit(EXIT_FAILURE);                          \
    }
#endif

// HIP Host functions to determine the gfx architecture
bool isGfx9()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx908") != std::string::npos)
            || (deviceName.find("gfx90a") != std::string::npos)
            || (deviceName.find("gfx940") != std::string::npos)
            || (deviceName.find("gfx941") != std::string::npos)
            || (deviceName.find("gfx942") != std::string::npos));
}

bool isGfx11()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx1100") != std::string::npos)
            || (deviceName.find("gfx1101") != std::string::npos)
            || (deviceName.find("gfx1102") != std::string::npos));
}

// HIP Host function to find if the device supports f64
uint32_t getGCNArchId()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);
    uint32_t mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_NONE;

    if(deviceName.find("gfx908") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX908;
    }
    else if(deviceName.find("gfx90a") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX90A;
    }
    else if(deviceName.find("gfx940") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX940;
    }
    else if(deviceName.find("gfx941") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX941;
    }
    else if(deviceName.find("gfx942") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX942;
    }
    else if(deviceName.find("gfx1100") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX1100;
    }
    else if(deviceName.find("gfx1101") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX1101;
    }
    else if(deviceName.find("gfx1102") != std::string::npos)
    {
        mGcnArch = rocwmma::Constants::AMDGCN_ARCH_ID_GFX1102;
    }

    return mGcnArch;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint32_t BlocksX,
          uint32_t BlocksY,
          uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t WaveSize,
          uint32_t ArchId>
bool checkConfigs()
{
    // Architecture we are testing
    const bool IsWave32 = (WaveSize == rocwmma::Constants::AMDGCN_WAVE_SIZE_32);
    const bool IsWave64 = (WaveSize == rocwmma::Constants::AMDGCN_WAVE_SIZE_64);

    const bool IsGfx908  = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX908);
    const bool IsGfx90A  = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX90A);
    const bool IsGfx940  = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX940);
    const bool IsGfx941  = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX941);
    const bool IsGfx942  = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX942);
    const bool IsGfx1100 = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX1100);
    const bool IsGfx1101 = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX1101);
    const bool IsGfx1102 = (ArchId == rocwmma::Constants::AMDGCN_ARCH_ID_GFX1102);

    const bool IsGfx9  = IsGfx908 || IsGfx90A || IsGfx940 || IsGfx941 || IsGfx942;
    const bool IsGfx11 = IsGfx1100 || IsGfx1101 || IsGfx1102;

    // Input Types testing
    const bool IsInputTInt8   = std::is_same_v<InputT, int8_t>;
    const bool IsInputTFloat8  = std::is_same_v<InputT, rocwmma::float8_t>;
    const bool IsInputTBFloat8 = std::is_same_v<InputT, rocwmma::bfloat8_t>;

    const bool IsInputTFloat16  = std::is_same_v<InputT, rocwmma::float16_t> || std::is_same_v<InputT, rocwmma::hfloat16_t>;
    const bool IsInputTBFloat16 = std::is_same_v<InputT, rocwmma::bfloat16_t>;

    const bool IsInputTFloat32  = std::is_same_v<InputT, rocwmma::float32_t>;
    const bool IsInputTXFloat32 = std::is_same_v<InputT, rocwmma::xfloat32_t>;

    const bool IsInputTFloat64 = std::is_same_v<InputT, rocwmma::float64_t>;

    // Output Types testing
    const bool IsOutputTInt8    = std::is_same_v<OutputT, int8_t>;
    const bool IsOutputTFloat8  = std::is_same_v<OutputT, rocwmma::float8_t>;
    const bool IsOutputTBFloat8 = std::is_same_v<OutputT, rocwmma::bfloat8_t>;

    const bool IsOutputTFloat16  = std::is_same_v<OutputT, rocwmma::float16_t> || std::is_same_v<OutputT, rocwmma::hfloat16_t>;
    const bool IsOutputTBFloat16 = std::is_same_v<OutputT, rocwmma::bfloat16_t>;

    const bool IsOutputTFloat32  = std::is_same_v<OutputT, rocwmma::float32_t>;
    const bool IsOutputTXFloat32 = std::is_same_v<OutputT, rocwmma::xfloat32_t>;

    const bool IsOutputTFloat64 = std::is_same_v<OutputT, rocwmma::float64_t>;

    // Compute Types testing
    const bool IsComputeTInt8    = std::is_same_v<ComputeT, int8_t>;
    const bool IsComputeTFloat8  = std::is_same_v<ComputeT, rocwmma::float8_t>;
    const bool IsComputeTBFloat8 = std::is_same_v<ComputeT, rocwmma::bfloat8_t>;

    const bool IsComputeTFloat16 = std::is_same_v<ComputeT, rocwmma::float16_t> || std::is_same_v<ComputeT, rocwmma::hfloat16_t>;
    const bool IsComputeTBFloat16 = std::is_same_v<ComputeT, rocwmma::bfloat16_t>;

    const bool IsComputeTFloat32  = std::is_same_v<ComputeT, rocwmma::float32_t>;
    const bool IsComputeTXFloat32 = std::is_same_v<ComputeT, rocwmma::xfloat32_t>;

    const bool IsComputeTFloat64 = std::is_same_v<ComputeT, rocwmma::float64_t>;

    // Block size testing
    const bool isBlockMN16 = (BlockM == 16u) && (BlockN == 16u);
    const bool isBlockMN32 = (BlockM == 32u) && (BlockN == 32u);

    // ThreadblockX must be a multiple of the wave size
    const bool TBlockXTest = (TBlockX % WaveSize == 0u);

    // Ensure that we have at least 1 wave
    const bool MinTBlockTest = (TBlockX >= WaveSize && TBlockY >= 1);

    // Ensure that we only build for the current compiler target, which
    // will also exclude the host.
    const bool CurrentArchTest = (ArchId == rocwmma::Constants::AMDGCN_CURRENT_ARCH_ID);

    // Ensure that we only build for the current wave size
    const bool CurrentWaveSizeTest = (WaveSize == rocwmma::Constants::AMDGCN_WAVE_SIZE);

    // Only supported hardware allowed
    const bool ArchTest = (bool)IsGfx9 || (bool)IsGfx11;

    // During the build phase, we have information about current target arch.
    // This means only the current arch and wave size are valid.
    const bool EnableBuild
        = (TBlockXTest && MinTBlockTest && CurrentArchTest && CurrentWaveSizeTest && ArchTest);

    // During run phase on the host, we don't have compile time info about current arch or wave size.
    // We have to trust that the runtime params obtained through HipDevice will dispatch correctly for
    // the current arch and wave size.
    const bool EnableRun = (TBlockXTest && MinTBlockTest && ArchTest);

    const bool EnableGfx9 = []() {
        const bool ArchTestGfx9 = (bool)IsGfx9;

        const bool WaveSizeTest = (bool)IsWave64;

        const bool TBlockTest
        = (TBlockX * TBlockY >= rocwmma::Constants::AMDGCN_WAVE_SIZE_64) && (TBlockX * TBlockY <= 256u);

        const bool InputTypesTest
        = (bool)IsInputTFloat8 || (bool)IsInputTBFloat8
          || (bool)IsInputTInt8 || (bool)IsInputTFloat16
          || (bool)IsInputTBFloat16 || (bool)IsInputTFloat32
          || (bool)IsInputTXFloat32 || (bool)IsInputTFloat64;

        // Gfx940/1/2 arch req'd for float8_t, bfloat8_t and xfloat32_t
        const bool F8XF32ArchTest
        = !((bool)IsInputTFloat8 || (bool)IsInputTBFloat8
          || (bool)IsInputTXFloat32)
          || (bool)IsGfx940 || (bool)IsGfx941
          || (bool)IsGfx942;

        // All archs except gfx908 can run float64_t
        const bool F64ArchTest
        = !(bool)IsInputTFloat64 || !(bool)IsGfx908;

        // General int8_t block size
        // BlockM/N = 16; Block K >= 16
        // BlockM/N = 32; Block K >= 8
        const bool I8BlockSizeTest
        = !((bool)IsInputTInt8)
          || ((bool)isBlockMN16 && (BlockK >= 16u)
          && (BlockK % 16u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 8u)
          && (BlockK % 8u == 0u));

        // Follow-on to gfx940/1/2 int8_t.
        // BlockM/N = 16; Block K >= 32
        // BlockM/N = 32; Block K >= 16
        const bool Gfx940I8BlockSizeTest
        = !((bool)IsInputTInt8
          && ((bool)IsGfx940 || (bool)IsGfx941
          || (bool)IsGfx942))
          || ((bool)isBlockMN16 && (BlockK >= 32u)
          && (BlockK % 32u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 16u)
          && (BlockK % 16u == 0u));

        // General float8_t / bfloat8_t block size
        // BlockM/N = 16; Block K >= 32
        // BlockM/N = 32; Block K >= 16
        const bool F8BlockSizeTest
        = !((bool)IsInputTFloat8 || (bool)IsInputTBFloat8)
          || ((bool)isBlockMN16 && (BlockK >= 32u)
          && (BlockK % 32u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 16u)
          && (BlockK % 16u == 0u));

        // General float16_t / hfloat16_t / bfloat16_t block size
        // BlockM/N = 16; Block K >= 16
        // BlockM/N = 32; Block K >= 8
        const bool F16BlockSizeTest
        = !((bool)IsInputTFloat16 || (bool)IsInputTBFloat16)
          || ((bool)isBlockMN16 && (BlockK >= 16u)
          && (BlockK % 16u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 8u)
          && (BlockK % 8u == 0u));

        // Older gfx908 arch has half BlockK on bfloat16_t
        // BlockM/N = 16; Block K >= 8
        // BlockM/N = 32; Block K >= 4
        const bool Gfx908BF16BlockSizeTest
        = !((bool)IsInputTBFloat16 && (bool)IsGfx908)
          || ((bool)isBlockMN16 && (BlockK >= 8u)
          && (BlockK % 8u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 4u)
          && (BlockK % 4u == 0u));

        // General float32_t block size
        // BlockM/N = 16; Block K >= 4
        // BlockM/N = 32; Block K >= 2
        const bool F32BlockSizeTest
        = !((bool)IsInputTFloat32)
          || ((bool)isBlockMN16 && (BlockK >= 4u)
          && (BlockK % 4u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 2u)
          && (BlockK % 2u == 0u));

        // General xfloat32_t block size
        // BlockM/N = 16; Block K >= 8
        // BlockM/N = 32; Block K >= 4
        const bool XF32BlockSizeTest
        = !((bool)IsInputTXFloat32)
          || ((bool)isBlockMN16 && (BlockK >= 8u)
          && (BlockK % 8u == 0u))
          || ((bool)isBlockMN32 && (BlockK >= 4u)
          && (BlockK % 4u == 0u));

        // General float64_t block size
        // BlockM/N = 16; Block K >= 4
        const bool F64BlockSizeTest
        = !((bool)IsInputTFloat64)
          || ((bool)isBlockMN16 && (BlockK >= 4u)
          && (BlockK % 4u == 0u));

        return (ArchTestGfx9 && WaveSizeTest && TBlockTest && InputTypesTest && F8XF32ArchTest
                    && F64ArchTest && I8BlockSizeTest && Gfx940I8BlockSizeTest && F8BlockSizeTest
                    && F16BlockSizeTest && Gfx908BF16BlockSizeTest && F32BlockSizeTest
                    && XF32BlockSizeTest && F64BlockSizeTest);
    };

    const bool EnableGfx11 = []() {

        // Valid for gfx11 only
        const bool ArchTestGfx11 = (bool)IsGfx11;

        // Wave size on gfx11 is 32
        const bool WaveSizeTest = (bool)IsWave32;

        // Max recommended TBlock size is 256
        const bool TBlockTest
        = (TBlockX * TBlockY >= rocwmma::Constants::AMDGCN_WAVE_SIZE_32) && (TBlockX * TBlockY <= 256u);

        // Input types supported
        const bool InputTypesTest = (bool)IsInputTInt8
                            || (bool)IsInputTFloat16
                            || (bool)IsInputTBFloat16;

        // General int8_t block size
        // BlockM/N = 16; Block K >= 16
        const bool I8BlockSizeTest = !((bool)IsInputTInt8)
                            || ((bool)isBlockMN16 && (BlockK >= 16u)
                                && (BlockK % 16u == 0u));

        // General float16_t / hfloat16_t / bfloat16_t block size
        // BlockM/N = 16; Block K >= 16
        const bool F16BlockSizeTest
        = !((bool)IsInputTFloat16 || (bool)IsInputTBFloat16)
            || ((bool)isBlockMN16 && (BlockK >= 16u)
                && (BlockK % 16u == 0u));

        return (ArchTestGfx11 && WaveSizeTest && TBlockTest && InputTypesTest && I8BlockSizeTest
                    && F16BlockSizeTest);

    };

    const bool finalEnableBuild = ((bool)EnableBuild
                                  && ((bool)EnableGfx9 || (bool)EnableGfx11));

    const bool finalEnableRun   = ((bool)EnableRun
                                  && ((bool)EnableGfx9 || (bool)EnableGfx11));

    return finalEnableBuild && finalEnableRun;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint32_t BlocksX,
          uint32_t BlocksY,
          uint32_t TBlockX,
          uint32_t TBlockY,
          uint32_t WaveSize>
const bool canEnable()
{
    uint32_t Id = getGCNArchId();
    // - Arch [gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1100, gfx1101, gfx1102]
    auto dispatchGuardFunc = [Id]() {
        bool dispatchResult = false;

#define ROCWMMA_CASE_BODY_ARG0(CASE_LABEL, CASE_IMPL) \
    case CASE_LABEL:                                  \
    {                                                 \
        CASE_IMPL                                     \
    }                                                 \
    break;

#define ROCWMMA_CASE_BODY_ARG1(CASE_LABEL, CASE_IMPL, CASE_IMPL_ARG0) \
    case CASE_LABEL:                                                  \
    {                                                                 \
        CASE_IMPL(CASE_IMPL_ARG0)                                     \
    }                                                                 \
    break;

#define ROCWMMA_SWITCH_BODY8_ARG1(SWITCH_ARG,                       \
                                  CASE_IMPL,                        \
                                  CASE_LABEL0,                      \
                                  CASE_LABEL1,                      \
                                  CASE_LABEL2,                      \
                                  CASE_LABEL3,                      \
                                  CASE_LABEL4,                      \
                                  CASE_LABEL5,                      \
                                  CASE_LABEL6,                      \
                                  CASE_LABEL7)                      \
    switch(SWITCH_ARG)                                              \
    {                                                               \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL0, CASE_IMPL, CASE_LABEL0) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL1, CASE_IMPL, CASE_LABEL1) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL2, CASE_IMPL, CASE_LABEL2) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL3, CASE_IMPL, CASE_LABEL3) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL4, CASE_IMPL, CASE_LABEL4) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL5, CASE_IMPL, CASE_LABEL5) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL6, CASE_IMPL, CASE_LABEL6) \
        ROCWMMA_CASE_BODY_ARG1(CASE_LABEL7, CASE_IMPL, CASE_LABEL7) \
    default:;                                                       \
    }


#define CASE_IMPL_ASSIGN1(ARCH_ID) \
dispatchResult = checkConfigs<BlockM, BlockN, BlockK, InputT, OutputT, ComputeT, BlocksX, BlocksY, TBlockX, TBlockY, WaveSize, ARCH_ID>();

#define DISPATCH_GUARD_BODY                     \
ROCWMMA_SWITCH_BODY8_ARG1(Id,                   \
                          CASE_IMPL_ASSIGN1,    \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX908,     \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX90A,     \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX940,     \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX941,     \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX942,     \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX1100,    \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX1101,    \
                          rocwmma::Constants::AMDGCN_ARCH_ID_GFX1102)

        DISPATCH_GUARD_BODY

#undef CASE_IMPL_ASSIGN1
#undef DISPATCH_GUARD_BODY

        return dispatchResult;
    };

    // Finally, execute and return the dispatch guard result
    return dispatchGuardFunc();
}

inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
}

inline double calculateTFlopsPerSec(
    uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs, uint32_t repeats = 1u)
{
    // elapsedTimeMs is over all iterations
    return calculateGFlops(m, n, k) / elapsedTimeMs * static_cast<double>(repeats);
}

// HIP Host function to retrieve the warp size
enum hipWarpSize_t : uint32_t
{
    Wave32 = 32,
    Wave64 = 64,
    UNSUPPORTED_WARP_SIZE,
};

uint32_t getWarpSize()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;
    uint32_t        mWarpSize = hipWarpSize_t::UNSUPPORTED_WARP_SIZE;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    switch(mProps.warpSize)
    {
    case hipWarpSize_t::Wave32:
    case hipWarpSize_t::Wave64:
        mWarpSize = mProps.warpSize;
    default:;
    }

    if(mWarpSize == hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
    {
        std::cerr << "Cannot proceed: unsupported warp sizev detected. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    return mWarpSize;
}

// Batched matrix data initialization
template <typename DataT>
__host__ static inline void
    fill(DataT* mat, uint32_t m, uint32_t k, uint32_t b, uint32_t normalization = 1)
{
    auto batchOffset = m * k;
    for(int t = 0; t < b; ++t)
    {
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                // Random values normalized such that output is between 0 and 1
                auto value = __float2half(static_cast<float>(rand() / normalization)
                                          / static_cast<float>(RAND_MAX));
                mat[t * batchOffset + i * k + j] = static_cast<DataT>(value);
            }
        }
    }
}

// Host matrix data random initialization
template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };
    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

// Host GEMM validation
template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t       m,
                         uint32_t       n,
                         uint32_t       k,
                         InputT const*  a,
                         InputT const*  b,
                         OutputT const* c,
                         OutputT*       d,
                         uint32_t       lda,
                         uint32_t       ldb,
                         uint32_t       ldc,
                         uint32_t       ldd,
                         ComputeT       alpha,
                         ComputeT       beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto cIndex = std::is_same<LayoutC, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, rocwmma::row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
#pragma omp parallel for
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

// Element-wise comparison
template <typename DataT>
__host__ std::pair<bool, double>
         compareEqual(DataT const* a, DataT const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        auto valA = a[i];
        auto valB = b[i];

        auto numerator = fabs(toDouble(valA) - toDouble(valB));
        auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = size;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

#endif // ROCWMMA_SAMPLES_COMMON_HPP
