#ifndef WMMA_IO_BROADCAST_H
#define WMMA_IO_BROADCAST_H

#include "IOPack.h"
#include "Types.h"
#include "Utils.h"

template<typename DataT>
struct PackedBroadcast;

template<>
struct PackedBroadcast<float16_t>
{
    using Traits = Pack<float16_t>::Traits;
    
    __device__ static inline auto exec(float16_t input) -> typename Traits::OutputT
    {  
        using OutputT = typename Traits::OutputT;
        using UnpackedT = typename Traits::UnpackedT;

        VecT<UnpackedT, 2> f16Vec;
        f16Vec[0] = input;
        f16Vec[1] = input;

        return OutputT(*reinterpret_cast<typename OutputT::VecType*>(&(f16Vec.v())));
    }
};

template<>
struct PackedBroadcast<float32_t>
{
    using Traits = Pack<float32_t>::Traits;
    
    __device__ static inline auto exec(float32_t input) -> typename Traits::OutputT
    {  
        return typename Traits::OutputT(input);
    }
};

template<typename DataT, uint32_t PackedRegisterCount>
struct PackedBroadcastRegs
{
    struct Traits
    {
        using Broadcaster = PackedBroadcast<DataT>;
        using PackedT = PackedType<DataT>;
        using OutputT = VecT<PackedT, PackedRegisterCount>;
    };

    __device__ static inline auto exec(DataT input) -> typename Traits::OutputT
    {
        using Broadcaster = typename Traits::Broadcaster;
        using OutputT = typename Traits::OutputT;

        OutputT result;
        auto it = result.begin();

        static_assert(decltype(it)::Range == PackedRegisterCount, "Iterator range doesn't match packed register count");

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