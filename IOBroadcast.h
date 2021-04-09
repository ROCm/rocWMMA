#ifndef WMMA_IO_BROADCAST_H
#define WMMA_IO_BROADCAST_H

#include "IOPack.h"
#include "Types.h"
#include "Utils.h"

template <typename DataT>
struct PackedBroadcast;

template <>
struct PackedBroadcast<float16_t>
{
    using Packer = Pack<float16_t, 2>;
    using Traits = typename Packer::Traits;

    __device__ static inline auto exec(float16_t input) -> typename Traits::OutputT
    {
        using InputT = typename Traits::InputT;

        InputT f16Vec;
        f16Vec[0] = input;
        f16Vec[1] = input;

        return Packer::exec(f16Vec);
    }
};

template <>
struct PackedBroadcast<hfloat16_t>
{
    using Packer = Pack<hfloat16_t, 2>;
    using Traits = typename Packer::Traits;

    __device__ static inline auto exec(hfloat16_t input) -> typename Traits::OutputT
    {
        using InputT = typename Traits::InputT;

        InputT h16Vec;
        h16Vec[0] = input;
        h16Vec[1] = input;

        return Packer::exec(h16Vec);
    }
};

template <>
struct PackedBroadcast<bfloat16_t>
{
    using Packer = Pack<bfloat16_t, 2>;
    using Traits = typename Packer::Traits;

    __device__ static inline auto exec(bfloat16_t input) -> typename Traits::OutputT
    {
        using InputT = typename Traits::InputT;

        InputT bf16Vec;
        bf16Vec[0] = input;
        bf16Vec[1] = input;

        return Packer::exec(bf16Vec);
    }
};

template <>
struct PackedBroadcast<float32_t>
{
    using Packer = Pack<float32_t, 1>;
    using Traits = typename Packer::Traits;

    __device__ static inline auto exec(float32_t input) -> typename Traits::OutputT
    {
        using InputT = typename Traits::InputT;
        return Packer::exec(InputT(input));
    }
};

template <typename DataT, uint32_t PackedRegisterCount>
struct PackedBroadcastRegs
{
    using Broadcaster = PackedBroadcast<DataT>;
    using Traits =
        typename Pack<DataT, PackedRegisterCount * Broadcaster::Traits::PackRatio>::Traits;

    __device__ static inline auto exec(DataT input) -> typename Traits::OutputT
    {
        using OutputT = typename Traits::OutputT;

        OutputT result;
        auto    it = result.begin();

        static_assert(decltype(it)::Range == PackedRegisterCount,
                      "Iterator range doesn't match packed register count");

#pragma unroll
        for(int i = 0; i < PackedRegisterCount; ++i)
        {
            *it = *Broadcaster::exec(input);
            it++;
        }
        return result;
    }
};

#endif // WMMA_IO_BROADCAST_H
