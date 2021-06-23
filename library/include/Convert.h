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
