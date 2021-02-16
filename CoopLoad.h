#ifndef WMMA_COOP_LOAD_H
#define WMMA_COOP_LOAD_H

#include "Types.h"
#include "Layout.h"
#include "LocalLoad.h"
#include "LocalStore.h"

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t WaveRows, uint32_t WaveCols>
struct amdgcn_cooperative_load_dword_traits
{
    // LDS set to row_major for performance
    using LdsDataFmt = row_major;

    // Result same format as buffer / local load
    using ResultT = typename amdgcn_buffer_load_dword_traits<BlockDim, BlockK, DataT>::ResultT;

    enum : uint32_t 
    {
        // Matrix A splits K direction loads amongst neighbouring waves in X, or column direction
        // Matrix B splits K direction loads amongst neighbouring waves in Y, or row direction
        SplitCount = std::is_same<MatrixT, matrix_a>::value ? WaveCols : WaveRows,

        // Matrix A will store blockDim.y blocks due to splitting in only X, or column direction
        // Matrix B will store blockDim.x / 64 blocks due to splitting in only Y, or row direction
        LdsBlockCount = std::is_same<MatrixT, matrix_a>::value ? WaveRows : WaveCols,

        // Statically calculate how much LDS mem is used.
        LdsBytes = BlockDim * BlockK * LdsBlockCount * sizeof(DataT),
    };

    static_assert(WaveRows > 0, "Wave row count must be greater than 0");
    static_assert(WaveCols > 0, "Wave col count must be greater than 0");
    static_assert(BlockK % SplitCount == 0, "BlockK size is not divisible by SplitCount");
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout, uint32_t WaveRows = 0, uint32_t WaveCols = 0>
struct amdgcn_cooperative_load_dword_DxK
{
    using Traits = amdgcn_cooperative_load_dword_traits<MatrixT, BlockDim, BlockK, DataT, WaveRows, WaveCols>;
    
    __device__ static auto exec(DataT const* globalPtr, uint32_t ldg, DataT* localPtr, uint32_t ldl) -> typename Traits::ResultT
    {     
        // Obtain the grid coordinate for splitting in the K direction.
        using MappingUtil = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
        auto gridCoord = MappingUtil::gridCoord();
        
        // Splitting the K direction:
        // Matrix A will split work with waves on same row (diff col)
        // Matrix B will split work with waves on same col (diff row)
        uint32_t gridIdSplit = (std::is_same<MatrixT, matrix_a>::value ? std::get<1>(gridCoord) : std::get<0>(gridCoord)) % Traits::SplitCount;

        // Global load using buffer.
        // Base address is the same, and split load by (SplitCount).
        // Multiply the gridId by the split load count to get iterative offset per wave.
        using GlobalBufferLoad = amdgcn_buffer_load_dword_DxK<MatrixT, BlockDim, BlockK / Traits::SplitCount, DataT, DataLayout>;
        using GlobalLoadLayout = typename GlobalBufferLoad::LayoutT;
        auto globalLoadOffset = GlobalLoadLayout::iterativeOffset(gridIdSplit * GlobalBufferLoad::Traits::LoadCount, ldg);
        auto splitLoad = GlobalBufferLoad::exec(globalPtr + globalLoadOffset, ldg);

        // Local store
        // Base offset is for threads working in another split group.
        using LocalStore = amdgcn_local_store_dword_DxK<MatrixT, BlockDim, BlockK / Traits::SplitCount, DataT, typename Traits::LdsDataFmt>;
        using LocalStoreLayout = typename LocalStore::LayoutT;
        auto ldsBaseOffset = BlockDim * BlockK * (std::is_same<MatrixT, matrix_a>::value ? threadIdx.y : (threadIdx.x / AMDGCN_WAVE_SIZE));
        auto localStoreOffset =  LocalStoreLayout::iterativeOffset(gridIdSplit * LocalStore::Traits::StoreCount, ldl);
        LocalStore::exec(localPtr + ldsBaseOffset + localStoreOffset, splitLoad, ldl);

        // Wait until all waves in the workgroup finish with their share of load.
        __syncthreads();

        // Perform the full load from LDS.
        using LocalLoad = amdgcn_local_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, typename Traits::LdsDataFmt>;
        return LocalLoad::exec(localPtr + ldsBaseOffset, ldl);
    }
};


// Wrapper for runtime wave count
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, 0, 0>
{
    template<uint32_t WaveRows>
    using CooperativeLoad = amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, WaveRows, 0>;

    // All loads will have the same result type
    struct Traits
    {
        using ResultT = typename CooperativeLoad<1>::Traits::ResultT;
    };

    __device__ static inline auto exec(DataT const* globalPtr, uint32_t ldg, DataT* localPtr, uint32_t ldl, uint32_t waveRows, uint32_t waveCols) -> typename Traits::ResultT
    {
        if(waveRows == 8)
        {
            return CooperativeLoad<8>::exec(globalPtr, ldg, localPtr, ldl, waveCols);
        }
        else if(waveRows == 4)
        {
            return CooperativeLoad<4>::exec(globalPtr, ldg, localPtr, ldl, waveCols);
        }
        else if(waveRows == 2)
        {
            return CooperativeLoad<2>::exec(globalPtr, ldg, localPtr, ldl, waveCols);
        }
        else if(waveRows == 1)
        {
            return CooperativeLoad<1>::exec(globalPtr, ldg, localPtr, ldl, waveCols);
        }
        else
        {
            assert(0 && "Unsupported wave col count");
            return typename Traits::ResultT();
        }
    }
};

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout, uint32_t WaveRows>
struct amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, WaveRows, 0>
{
    template<uint32_t WaveCols>
    using CooperativeLoad = amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, WaveRows, WaveCols>;
    
    // All loads will have the same result type
    struct Traits
    {
        using ResultT = typename CooperativeLoad<1>::Traits::ResultT;
    };

    __device__ static inline auto exec(DataT const* globalPtr, uint32_t ldg, DataT* localPtr, uint32_t ldl, uint32_t waveCols) -> typename Traits::ResultT
    {
        if(waveCols == 8)
        {
            return CooperativeLoad<8>::exec(globalPtr, ldg, localPtr, ldl);
        }
        else if(waveCols == 4)
        {
            return CooperativeLoad<4>::exec(globalPtr, ldg, localPtr, ldl);
        }
        else if(waveCols == 2)
        {
            return CooperativeLoad<2>::exec(globalPtr, ldg, localPtr, ldl);
        }
        else if(waveCols == 1)
        {
            return CooperativeLoad<1>::exec(globalPtr, ldg, localPtr, ldl);
        }
        else
        {
            assert(0 && "Unsupported wave col count");
            return typename Traits::ResultT();
        }
    }
};

#endif // WMMA_COOP_LOAD_H