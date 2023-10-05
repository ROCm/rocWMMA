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
#ifndef ROCWMMA_CONFIG_HPP
#define ROCWMMA_CONFIG_HPP

namespace rocwmma
{

///
/// Architecture support
/// Guaranteed symbols:
/// ROCWMMA_ARCH_GFX908
/// ROCWMMA_ARCH_GFX90a
/// ROCWMMA_ARCH_GFX940
/// ROCWMMA_ARCH_GFX941
/// ROCWMMA_ARCH_GFX942
/// ROCWMMA_ARCH_GFX1100
/// ROCWMMA_ARCH_GFX1101
/// ROCWMMA_ARCH_GFX1102
#if defined(__gfx908__)
#define ROCWMMA_ARCH_GFX908 __gfx908__
#elif defined(__gfx90a__)
#define ROCWMMA_ARCH_GFX90A __gfx90a__
#elif defined(__gfx940__)
#define ROCWMMA_ARCH_GFX940 __gfx940__
#elif defined(__gfx941__)
#define ROCWMMA_ARCH_GFX941 __gfx941__
#elif defined(__gfx942__)
#define ROCWMMA_ARCH_GFX942 __gfx942__
#elif defined(__gfx1100__)
#define ROCWMMA_ARCH_GFX1100 __gfx1100__
#elif defined(__gfx1101__)
#define ROCWMMA_ARCH_GFX1101 __gfx1101__
#elif defined(__gfx1102__)
#define ROCWMMA_ARCH_GFX1102 __gfx1102__
#else
#define ROCWMMA_ARCH_HOST 1
#endif

#if !defined(ROCWMMA_ARCH_GFX908)
#define ROCWMMA_ARCH_GFX908 0
#endif
#if !defined(ROCWMMA_ARCH_GFX90A)
#define ROCWMMA_ARCH_GFX90A 0
#endif
#if !defined(ROCWMMA_ARCH_GFX940)
#define ROCWMMA_ARCH_GFX940 0
#endif
#if !defined(ROCWMMA_ARCH_GFX941)
#define ROCWMMA_ARCH_GFX941 0
#endif
#if !defined(ROCWMMA_ARCH_GFX942)
#define ROCWMMA_ARCH_GFX942 0
#endif
#if !defined(ROCWMMA_ARCH_GFX1100)
#define ROCWMMA_ARCH_GFX1100 0
#endif
#if !defined(ROCWMMA_ARCH_GFX1101)
#define ROCWMMA_ARCH_GFX1101 0
#endif
#if !defined(ROCWMMA_ARCH_GFX1102)
#define ROCWMMA_ARCH_GFX1102 0
#endif
#if !defined(ROCWMMA_ARCH_HOST)
#define ROCWMMA_ARCH_HOST 0
#endif

///
/// Architecture configuration
/// Guaranteed symbols:
/// ROCWMMA_ARCH_GFX9
/// ROCWMMA_ARCH_GFX11
/// ROCWMMA_WAVE64_MODE
/// ROCWMMA_WAVE32_MODE
/// ROCWMMA_BLOCK_DIM_16_SUPPORTED
/// ROCWMMA_BLOCK_DIM_32_SUPPORTED
///
#if ROCWMMA_ARCH_GFX908 || ROCWMMA_ARCH_GFX90A || ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 \
    || ROCWMMA_ARCH_GFX942
#define ROCWMMA_ARCH_GFX9 1
#define ROCWMMA_WAVE64_MODE 1
#define ROCWMMA_BLOCK_DIM_16_SUPPORTED 1
#define ROCWMMA_BLOCK_DIM_32_SUPPORTED 1
#endif

#if ROCWMMA_ARCH_GFX1100 || ROCWMMA_ARCH_GFX1101 || ROCWMMA_ARCH_GFX1102
#define ROCWMMA_ARCH_GFX11 1
#define ROCWMMA_WAVE32_MODE 1
#define ROCWMMA_BLOCK_DIM_16_SUPPORTED 1
#endif

#if !defined(ROCWMMA_ARCH_GFX9)
#define ROCWMMA_ARCH_GFX9 0
#endif
#if !defined(ROCWMMA_ARCH_GFX11)
#define ROCWMMA_ARCH_GFX11 0
#endif
#if !defined(ROCWMMA_WAVE64_MODE)
#define ROCWMMA_WAVE64_MODE 0
#endif
#if !defined(ROCWMMA_WAVE32_MODE)
#define ROCWMMA_WAVE32_MODE 0
#endif
#if !defined(ROCWMMA_BLOCK_DIM_16_SUPPORTED)
#define ROCWMMA_BLOCK_DIM_16_SUPPORTED 0
#endif
#if !defined(ROCWMMA_BLOCK_DIM_32_SUPPORTED)
#define ROCWMMA_BLOCK_DIM_32_SUPPORTED 0
#endif

#if defined(NDEBUG)
#define ROCWMMA_UNSUPPORTED_IMPL(MSG)
#else
#define ROCWMMA_UNSUPPORTED_IMPL(MSG) __attribute__((deprecated(MSG)))
#endif

#if defined(HIP_NO_HALF)
#define ROCWMMA_NO_HALF 1
#else
#define ROCWMMA_NO_HALF 0
#endif // HIP_NO_HALF

#if ROCWMMA_NO_HALF || (!ROCWMMA_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__))
#define ROCWMMA_TESTS_NO_HALF 1
#else
#define ROCWMMA_TESTS_NO_HALF 0
#endif // !ROCWMMA_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__)

///
/// Sanity checks
///
#if ROCWMMA_ARCH_GFX11
    static_assert((bool)(ROCWMMA_WAVE32_MODE) && !(bool)(ROCWMMA_WAVE64_MODE),
                  "rocWMMA supports only wave32 for gfx11 arch");
    static_assert((bool)(ROCWMMA_BLOCK_DIM_16_SUPPORTED) && !(bool)(ROCWMMA_BLOCK_DIM_32_SUPPORTED),
                  "rocWMMA supports only block size of 16 for gfx11 arch");
#endif

#if ROCWMMA_ARCH_GFX9
    static_assert(!(bool)(ROCWMMA_WAVE32_MODE) && (bool)(ROCWMMA_WAVE64_MODE),
                  "rocWMMA supports only wave64 for gfx9 arch");
    static_assert((bool)(ROCWMMA_BLOCK_DIM_16_SUPPORTED) && (bool)(ROCWMMA_BLOCK_DIM_32_SUPPORTED),
                  "rocWMMA requires block size of 16 and 32 for gfx9 arch");
#endif

///
/// Host and Device symbols
///
#define ROCWMMA_DEVICE __device__

#define ROCWMMA_HOST __host__

#define ROCWMMA_HOST_DEVICE ROCWMMA_HOST ROCWMMA_DEVICE

#define ROCWMMA_KERNEL __global__

} // namespace rocwmma

#endif // ROCWMMA_CONFIG_HPP
