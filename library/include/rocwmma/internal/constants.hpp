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
#ifndef ROCWMMA_CONSTANTS_HPP
#define ROCWMMA_CONSTANTS_HPP

namespace rocwmma
{

    static constexpr uint32_t AMDGCN_WAVE_SIZE                   = 64u;
    static constexpr uint32_t AMDGCN_REGISTER_ELEMENT_SIZE_BYTES = 4u;
    static constexpr uint32_t AMDGCN_REGISTER_SIZE_BYTES
        = AMDGCN_REGISTER_ELEMENT_SIZE_BYTES * AMDGCN_WAVE_SIZE;

    static constexpr uint32_t AMDGCN_LDS_MAX_SIZE_BYTES    = 65536u;
    static constexpr uint32_t AMDGCN_CACHE_LINE_SIZE_BYTES = 64u;
    static constexpr uint32_t AMDGCN_DWORD_SIZE_BYTES      = 4u;

} // namespace rocwmma

#endif // ROCWMMA_CONSTANTS_HPP
