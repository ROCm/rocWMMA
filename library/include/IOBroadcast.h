#ifndef WMMA_IO_BROADCAST_H
#define WMMA_IO_BROADCAST_H

#include "IOPack.h"
#include "IOTraits.h"
#include "Types.h"
#include "Utils.h"

// Broadcast generates vectors of desired values
template <typename DataT, uint32_t VectorSize>
struct Broadcast
{
    struct Traits
    {
        using OutputT = VecT<DataT, VectorSize>;
    };

    __device__ static inline auto exec(DataT val) -> typename Traits::OutputT
    {
        using OutputT = typename Traits::OutputT;

        OutputT output;

#pragma unroll
        for(unsigned int i = 0; i < OutputT::size(); i++)
        {
            output[i] = val;
        }

        return output;
    }
};

#endif // WMMA_IO_BROADCAST_H
