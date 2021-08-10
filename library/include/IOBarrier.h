#ifndef WMMA_IO_BARRIER_H
#define WMMA_IO_BARRIER_H

//Perform synchronization across fragments(wavefronts) in a workgroup
struct amdgcn_barrier
{
    __device__ static inline auto exec() 
    {
        return __builtin_amdgcn_s_barrier();
    }
};

#endif // WMMA_IO_BARRIER_H
