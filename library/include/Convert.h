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
#ifndef WMMA_CONVERT_H
#define WMMA_CONVERT_H

#include "IOPack.h"
#include "Types.h"

template <typename InputT, typename OutputT>
struct amdgcn_convert
{
    template <uint32_t NumRegs>
    __device__ static inline auto exec(VecT<InputT, NumRegs> const& regsIn)
        -> VecT<OutputT, NumRegs>
    {
        VecT<OutputT, NumRegs> result;

#pragma unroll
        for(unsigned i = 0; i < NumRegs; i++)
        {
            result[i] = static_cast<OutputT>(regsIn[i]);
        }
        return result;
    }
};

template <typename T>
struct amdgcn_convert<T, T>
{
    template <typename IncomingT>
    __device__ static inline auto exec(IncomingT&& regsIn) -> IncomingT&&
    {
        return std::forward<IncomingT>(regsIn);
    }
};

template <>
struct amdgcn_convert<hfloat16_t, float32_t>
{
    template <uint32_t NumRegs>
    __device__ static inline auto exec(VecT<hfloat16_t, NumRegs> const& regsIn)
        -> VecT<float32_t, NumRegs>
    {
        VecT<float32_t, NumRegs> result;

#pragma unroll
        for(unsigned i = 0; i < NumRegs; i++)
        {
            result[i] = __half2float(regsIn[i]);
        }
        return result;
    }
};

template <>
struct amdgcn_convert<float32_t, hfloat16_t>
{
    template <uint32_t NumRegs>
    __device__ static inline auto exec(VecT<float32_t, NumRegs> const& regsIn)
        -> VecT<hfloat16_t, NumRegs>
    {
        VecT<hfloat16_t, NumRegs> result;

#pragma unroll
        for(unsigned i = 0; i < NumRegs; i++)
        {
            result[i] = __float2half(regsIn[i]);
        }
        return result;
    }
};

#endif // WMMA_CONVERT_H
