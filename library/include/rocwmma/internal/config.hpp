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
#ifndef ROCWMMA_CONFIG_HPP
#define ROCWMMA_CONFIG_HPP

#include <iostream>

namespace rocwmma
{
#if defined(__gfx908__)
#define ROCWMMA_ARCH_GFX908
#elif defined(__gfx90a__)
#define ROCWMMA_ARCH_GFX90A
#elif defined(__gfx1100__)
#define ROCWMMA_ARCH_GFX1100
#elif defined(__gfx1101__)
#define ROCWMMA_ARCH_GFX1101
#elif defined(__gfx1102__)
#define ROCWMMA_ARCH_GFX1102
#else
#define ROCWMMA_ARCH_NONE
#endif

#if defined(ROCWMMA_ARCH_GFX908) || defined(ROCWMMA_ARCH_GFX90A)
#define ROCWMMA_ARCH_MI
#define ROCWMMA_WAVE64_MODE
#define ROCWMMA_BLOCK_DIM_32_SUPPORTED
#elif defined(ROCWMMA_ARCH_GFX1100) || defined(ROCWMMA_ARCH_GFX1101) \
    || defined(ROCWMMA_ARCH_GFX1102)
#define ROCWMMA_ARCH_NAVI
#define ROCWMMA_WAVE32_MODE
#endif

#if defined(ROCWMMA_ARCH_NAVI) && defined(ROCWMMA_BLOCK_DIM_32_SUPPORTED)
#error " Navi / 32 Block dimensions are not supported together"
#endif

#if defined(ROCWMMA_ARCH_NAVI) && defined(ROCWMMA_WAVE64_MODE)
#error " Navi / 64 Wave mode are not supported together"
#endif

#if defined(ROCWMMA_ARCH_MI) && defined(ROCWMMA_WAVE32_MODE)
#error " MI / 32 Wave mode are not supported together"
#endif

} // namespace rocwmma

#endif // ROCWMMA_CONFIG_HPP
