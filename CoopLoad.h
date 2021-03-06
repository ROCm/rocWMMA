#ifndef WMMA_COOP_LOAD_H
#define WMMA_COOP_LOAD_H

#include "IOConfig.h"
#include "IOTraits.h"
#include "Layout.h"
#include "MappingUtil.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Types.h"

template <typename MatrixT,
          uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          uint32_t WaveRows = 0,
          uint32_t WaveCols = 0>
struct amdgcn_cooperative_load_dword_DxK
{
    struct Traits
    {
        enum : uint32_t
        {
            // Matrix A splits K direction loads amongst neighbouring waves in X, or column direction
            // Matrix B splits K direction loads amongst neighbouring waves in Y, or row direction
            SplitCount = std::is_same<MatrixT, matrix_a>::value ? WaveCols : WaveRows,

            // Matrix A will store blockDim.x / 64 blocks due competing waves in other rows
            // Matrix B will store blockDim.y blocks due to waves in other cols
            LdsBlockCount = std::is_same<MatrixT, matrix_a>::value ? WaveRows : WaveCols,

            // Statically calculate how much LDS mem is used.
            LdsBytes = BlockDim * BlockK * LdsBlockCount * sizeof(DataT),
        };

        static_assert(WaveRows > 0, "Wave row count must be greater than 0");
        static_assert(WaveCols > 0, "Wave col count must be greater than 0");
        static_assert(BlockK % SplitCount == 0, "BlockK size is not divisible by SplitCount");

        // LDS set to row_major for performance
        using LdsDataFmt = row_major;

        // Same unpacked register count throughout
        using OutputT
            = VecT<DataT, amdgcn_io_traits<BlockDim, BlockK, DataT>::UnpackedRegisterCount>;
    };

    __device__ static auto exec(DataT const* globalPtr, uint32_t ldg, DataT* localPtr, uint32_t ldl)
        -> typename Traits::OutputT
    {
        // Splitting the K direction:
        // Matrix A will share work with waves on same row (different col)
        // Matrix B will share work with waves on same col (different row)
        // Matrix A will compete for LDS with waves on different row
        // Matrix B will compete for LDS with waves on different col
        using MappingUtil     = MappingUtil<BlockDim, BlockK, DataT, DataLayout>;
        auto     waveCoord    = MappingUtil::waveCoord(); // Local to workgroup
        uint32_t sharedWaveId = (std::is_same<MatrixT, matrix_a>::value ? std::get<1>(waveCoord)
                                                                        : std::get<0>(waveCoord));
        uint32_t competingWaveId
            = (std::is_same<MatrixT, matrix_a>::value ? std::get<0>(waveCoord)
                                                      : std::get<1>(waveCoord));

        // Global load using buffer.
        // Base address is the same, and split load by (SplitCount).
        // Multiply the gridId by the split load count to get iterative offset per wave.
        using GlobalConfig
            = IOConfig<MatrixT, BlockDim, BlockK / Traits::SplitCount, DataT, DataLayout>;
        using GlobalLoad       = typename GlobalConfig::GlobalLoader;
        using GlobalLoadLayout = typename GlobalLoad::Traits::LayoutT;
        auto globalLoadOffset
            = GlobalLoadLayout::iterativeOffset(sharedWaveId * GlobalLoad::Traits::IOCount, ldg);
        auto splitLoad = GlobalLoad::exec(globalPtr + globalLoadOffset, ldg);

        // Local store
        // Base offset is for threads working in another split group.
        using LocalStoreConfig = IOConfig<MatrixT,
                                          BlockDim,
                                          BlockK / Traits::SplitCount,
                                          DataT,
                                          typename Traits::LdsDataFmt>;
        using LocalStore       = typename LocalStoreConfig::CoopLocalStorer;
        using LocalStoreLayout = typename LocalStore::Traits::LayoutT;
        auto ldsBaseOffset     = BlockDim * BlockK * competingWaveId;
        auto localStoreOffset  = LocalStoreLayout::iterativeOffset(
            sharedWaveId * LocalStoreConfig::IOTraits::IOCount, ldl);
        LocalStore::exec(localPtr + ldsBaseOffset + localStoreOffset, splitLoad, ldl);

        // Wait until all waves in the workgroup finish with their share of load.
        __syncthreads();

        // Perform the full block load from LDS.
        using LocalLoadConfig
            = IOConfig<MatrixT, BlockDim, BlockK, DataT, typename Traits::LdsDataFmt>;
        using LocalLoad = typename LocalLoadConfig::CoopLocalLoad;
        auto result     = LocalLoad::exec(localPtr + ldsBaseOffset, ldl);

        // Make sure that everything is read
        __syncthreads();

        return result;
    }
};

// Wrapper for runtime wave count
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, 0, 0>
{
    template <uint32_t WaveRows>
    using CooperativeLoad = amdgcn_cooperative_load_dword_DxK<MatrixT,
                                                              BlockDim,
                                                              BlockK,
                                                              DataT,
                                                              DataLayout,
                                                              WaveRows,
                                                              0>;

    // All loads will have the same result type
    struct Traits
    {
        using OutputT = typename CooperativeLoad<1>::Traits::OutputT;
    };

    __device__ static inline auto exec(DataT const* globalPtr,
                                       uint32_t     ldg,
                                       DataT*       localPtr,
                                       uint32_t     ldl,
                                       uint32_t     waveRows,
                                       uint32_t     waveCols) -> typename Traits::OutputT
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
            return typename Traits::OutputT();
        }
    }
};

template <typename MatrixT,
          uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          uint32_t WaveRows>
struct amdgcn_cooperative_load_dword_DxK<MatrixT, BlockDim, BlockK, DataT, DataLayout, WaveRows, 0>
{
    template <uint32_t WaveCols>
    using CooperativeLoad = amdgcn_cooperative_load_dword_DxK<MatrixT,
                                                              BlockDim,
                                                              BlockK,
                                                              DataT,
                                                              DataLayout,
                                                              WaveRows,
                                                              WaveCols>;

    // All loads will have the same result type
    struct Traits
    {
        using OutputT = typename CooperativeLoad<1>::Traits::OutputT;
    };

    __device__ static inline auto
        exec(DataT const* globalPtr, uint32_t ldg, DataT* localPtr, uint32_t ldl, uint32_t waveCols)
            -> typename Traits::OutputT
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
            return typename Traits::OutputT();
        }
    }
};

#endif // WMMA_COOP_LOAD_H
