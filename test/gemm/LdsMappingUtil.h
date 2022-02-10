#ifndef LDS_MAPPING_UTIL_H
#define LDS_MAPPING_UTIL_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/WMMA.h>
#include <rocwmma/WMMACoop.h>
#pragma GCC diagnostic pop

#include <rocwmma/internal/IOConfig.h>
#include <rocwmma/internal/MappingUtil.h>
#include <rocwmma/internal/Utils.h>

namespace rocwmma
{
    class LdsKW
    {
    };
    class LdsKH
    {
    };
    class LdsKHC
    {
    };
    class LdsRF
    {
    };

    template <>
    constexpr const char* dataTypeToString<LdsKW>()
    {
        return "LdsKW";
    }

    template <>
    constexpr const char* dataTypeToString<LdsKH>()
    {
        return "LdsKH";
    }

    template <>
    constexpr const char* dataTypeToString<LdsKHC>()
    {
        return "LdsKHC";
    }

    template <>
    constexpr const char* dataTypeToString<LdsRF>()
    {
        return "LdsRF";
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              typename MappingLds,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil;

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil<BlockM,
                          BlockN,
                          BlockK,
                          DataT,
                          LayoutA,
                          LayoutB,
                          LayoutLds,
                          LdsRF,
                          BlocksX,
                          BlocksY>
    {
        /* LdsRF (Register file)
    * Lds usage is mapped as a vertical register file of width 64.
    * This geometry is compatible with both A and B fragments, as
    * fragments do not support partial registers, regardless of
    * DataLayout.
    *
    * There is no need for data transpose as A and B should already
    * be aligned.
    *
    * All that really needs to happen is we interpret the fragments
    * as registers and provide 1-1 mapping into LDS memory with width
    * of 64. The matrix_b type of LDS fragments is conducive to this
    * mapping as the current config writes rows of fragment data, such
    * as a vertical register file would need.
    *
    * In this case, block dimensions for A and B cannot be extended by
    * respective BlocksX and BlocksY, due to the LDS geometry which does
    * not easily map to sub-blocks.
    *
    * Local Layout (LDS): e.g. BlockMNK = (32, 32, 8)
    *
    *           Elements 0.....31 32.....64
    *                    _______________
    *           Reg0    |  C0   |   C4  |
    *  BlockA   Reg1    |  C1   |   C5  |
    *    (0)    Reg2    |  C2   |   C6  |
    *           Reg3    |  C3   |   C7  |
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  C0   |   C4  |
    *  BlockA   Reg1    |  C1   |   C5  |
    *   (X-1)   Reg2    |  C2   |   C6  |
    *           Reg3    |  C3   |   C7  |
    *
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  R0   |   R4  |
    *  BlockB   Reg1    |  R1   |   R5  |
    *    (0)    Reg2    |  R2   |   R6  |
    *           Reg3    |  R3   |   R7  |
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  R0   |   R4  |
    *  BlockB   Reg1    |  R1   |   R5  |
    *   (Y-1)   Reg2    |  R2   |   R6  |
    *           Reg3    |  R3   |   R7  |
    */

        static constexpr uint32_t registerFileWidth = AMDGCN_WAVE_SIZE;

        // Global read - individual block size
        using GlobalReadFragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
        using GlobalReadFragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        // Local Write
        // Vertical register file fulfilled by matrix_b with BlockN = 64
        using LocalWriteFragA
            = fragment<matrix_b, 1, registerFileWidth, GlobalReadFragA::size(), DataT, LayoutLds>;
        using LocalWriteFragB
            = fragment<matrix_b, 1, registerFileWidth, GlobalReadFragB::size(), DataT, LayoutLds>;

        // Sanity checks
        static_assert(GlobalReadFragA::size() == LocalWriteFragA::size(),
                      "Incompatible register count");
        static_assert(GlobalReadFragB::size() == LocalWriteFragB::size(),
                      "Incompatible register count");
        static_assert(LocalWriteFragA::size() * registerFileWidth == BlockM * BlockK,
                      "Incompatible element count");
        static_assert(LocalWriteFragB::size() * registerFileWidth == BlockN * BlockK,
                      "Incompatible element count");

        // Local read same as write
        using LocalReadFragA = LocalWriteFragA;
        using LocalReadFragB = LocalWriteFragB;

        // Final fragments are same as global
        using FragA = GlobalReadFragA;
        using FragB = GlobalReadFragB;

    private:
        // Calculate offsets based on DataLayout of LDS, A frags and B frags.
        // All of them will have the same workgroupDim and waveCoord values.
        using LdsOffsets     = typename LocalWriteFragA::IOConfig::MappingUtil;
        using GlobalAOffsets = typename GlobalReadFragA::IOConfig::MappingUtil;
        using GlobalBOffsets = typename GlobalReadFragB::IOConfig::MappingUtil;

    public:
        __device__ static inline auto ldsWidth()
        {
            return registerFileWidth;
        }

        __device__ static inline auto ldsHeight()
        {
            auto workgroupDim = LdsOffsets::workgroupDim();
            return LocalWriteFragA::size() * BlocksX * std::get<0>(workgroupDim)
                   + LocalWriteFragB::size() * BlocksY * std::get<1>(workgroupDim);
        }

        __device__ static inline uint32_t ld()
        {
            return std::is_same<LayoutLds, row_major>::value ? ldsWidth() : ldsHeight();
        }

        __device__ static inline uint32_t baseOffsetA()
        {
            return 0;
        }

        __device__ static inline uint32_t baseOffsetB()
        {
            auto matrixCoord = std::make_pair(
                LocalWriteFragA::size() * BlocksX * std::get<0>(LdsOffsets::workgroupDim()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetA()
        {
            auto matrixCoord = std::make_pair(
                LocalWriteFragA::size() * BlocksX * std::get<0>(LdsOffsets::waveCoord()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetB()
        {
            auto matrixCoord = std::make_pair(
                LocalWriteFragB::size() * BlocksY * std::get<1>(LdsOffsets::waveCoord()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
        {
            auto matrixCoord = std::make_pair(LocalWriteFragA::size() * blockX, 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
        {
            auto matrixCoord = std::make_pair(LocalWriteFragB::size() * blockY, 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline void
            prefetchGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = GlobalAOffsets::workgroupDim();
            auto           waveCoord    = GlobalAOffsets::waveCoord();
            auto           waveIndex    = std::get<1>(waveCoord);
            auto           waveCount    = std::get<1>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            for(int32_t i = 0; i < BlocksX; ++i)
            {
                // Issue global read
                GlobalReadFragA fetchA;
                load_matrix_coop_sync(
                    fetchA,
                    baseA + GlobalAOffsets::dataOffset(lda, std::make_pair(BlockM * i, 0)),
                    lda,
                    waveIndex,
                    waveCount,
                    splitCount);

                // Issue local store
                store_matrix_coop_sync(baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(i),
                                       reinterpret_cast<LocalWriteFragA&>(fetchA),
                                       ld(),
                                       waveIndex,
                                       waveCount,
                                       splitCount);
            }
        }

        __device__ static inline void
            prefetchGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = GlobalBOffsets::workgroupDim();
            auto           waveCoord    = GlobalBOffsets::waveCoord();
            auto           waveIndex    = std::get<0>(waveCoord);
            auto           waveCount    = std::get<0>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            for(int32_t i = 0; i < BlocksY; ++i)
            {
                // Issue global read
                GlobalReadFragB fetchB;
                load_matrix_coop_sync(
                    fetchB,
                    baseB + GlobalBOffsets::dataOffset(ldb, std::make_pair(0, BlockN * i)),
                    ldb,
                    waveIndex,
                    waveCount,
                    splitCount);

                // Issue local store
                store_matrix_coop_sync(baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(i),
                                       reinterpret_cast<LocalWriteFragB&>(fetchB),
                                       ld(),
                                       waveIndex,
                                       waveCount,
                                       splitCount);
            }
        }

        __device__ static inline void
            prefetchLocalA(FragA& fragA, DataT const* baseLds, uint32_t blockX)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragA&>(fragA),
                             baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(blockX),
                             ld());
        }

        __device__ static inline void
            prefetchLocalB(FragB& fragB, DataT const* baseLds, uint32_t blockY)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragB&>(fragB),
                             baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(blockY),
                             ld());
        }

        __device__ static inline void prefetchLocalA(FragA* fragsA, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                prefetchLocalA(fragsA[i], baseLds, i);
            }
        }

        __device__ static inline void prefetchLocalB(FragB* fragsB, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                prefetchLocalB(fragsB[i], baseLds, i);
            }
        }
    };

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil<BlockM,
                          BlockN,
                          BlockK,
                          DataT,
                          LayoutA,
                          LayoutB,
                          LayoutLds,
                          LdsKW,
                          BlocksX,
                          BlocksY>
    {
        /* LdsKW (BlockK = LDS Width)
    * Matrix geometry for A and B have a common dimension (BlockK).
    * We can fix one of the LDS dimensions to BlockK (in this case the width),
    * and insert blocks of different heights (BlockM, BlockN) to use the space
    * without the need of extra padding.
    *
    * This format is naturally conducive to matrix_a layout, whereas matrix_b
    * blocks are transposed.
    *
    * In this case, block dimensions for A and B are extended by
    * respective BlocksX and BlocksY, this allows for more efficient global
    * loads. These dimensions are also preserved in LDS, in which smaller
    * BlockMNK can be easily indexed.
    *
    * Local Layout (LDS):
    *
    *                      _____________BlockK____________
    *                     |                               |
    *                     v                               v
    *                     kDim --------------------------->
    *                   -->______________    ...        ____
    *                   |  |    |    |                  |    |
    *                   |  |    |    |                  |    |
    * (BlockM * BlocksX)|  | C0 | C1 | C2               | Ck |   A
    *                   |  |    |    |                  |    |
    *                   |  |___ |___ |____    ...       |____|
    *                   -->
    *                       ...  ...  ...    ...         ...
    *
    *                   -->______________    ...        ____
    *                   |  |    |    |                  |    |
    *                   |  |    |    |                  |    |
    * (BlockN * BlocksY)|  | R0 | R1 | R2               | Rk |   B (T)
    *                   |  |    |    |                  |    |
    *                   |  |___ |___ |____    ...       |____|
    *                   -->
    *                       ...  ...  ...    ...         ...
    */

        // Global read
        // A matrix extends across BlocksX
        // B matrix extends across BlocksY
        using GlobalReadFragA
            = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, DataT, LayoutA>;
        using GlobalReadFragB
            = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, DataT, LayoutB>;

        // Local Write
        // A matrix already aligned
        // B matrix is transposed
        using LocalWriteFragA
            = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, DataT, LayoutLds>;
        using LocalWriteFragB = fragment<matrix_a,
                                         BlockN * BlocksY,
                                         BlockM,
                                         BlockK,
                                         DataT,
                                         LayoutLds>; // Exchange rows / cols

        // Local Read - smaller block size target loads
        // A matrix reads BlockM x BlockK
        // B matrix reads BlockK x BlockN (transposed)
        using LocalReadFragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutLds>;
        using LocalReadFragB = fragment<matrix_a, BlockN, BlockM, BlockK, DataT, LayoutLds>;

        // Final in-register fragment from LDS load
        using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
        using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        using MappingUtil = MappingUtil<BlockM, BlockN, DataT, LayoutLds>;

        __device__ static inline auto ldsWidth()
        {
            return BlockK;
        }

        __device__ static inline auto ldsHeight()
        {
            auto workgroupDim = MappingUtil::workgroupDim();
            return BlockM * BlocksX * std::get<0>(workgroupDim)
                   + BlockN * BlocksY * std::get<1>(workgroupDim);
        }

        __device__ static inline uint32_t ld()
        {
            return std::is_same<LayoutLds, row_major>::value ? ldsWidth() : ldsHeight();
        }

        __device__ static inline uint32_t baseOffsetA()
        {
            return 0;
        }

        __device__ static inline uint32_t baseOffsetB()
        {
            auto matrixCoord
                = std::make_pair(BlockM * BlocksX * std::get<0>(MappingUtil::workgroupDim()), 0);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetA()
        {
            auto matrixCoord
                = std::make_pair(BlockM * BlocksX * std::get<0>(MappingUtil::waveCoord()), 0);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetB()
        {
            auto matrixCoord
                = std::make_pair(BlockN * BlocksY * std::get<1>(MappingUtil::waveCoord()), 0);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
        {
            auto matrixCoord = std::make_pair(BlockM * blockX, 0);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
        {
            auto matrixCoord = std::make_pair(BlockN * blockY, 0);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline void
            prefetchGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = MappingUtil::workgroupDim();
            auto           waveCoord    = MappingUtil::waveCoord();
            auto           waveIndex    = std::get<1>(waveCoord);
            auto           waveCount    = std::get<1>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Issue global read
            GlobalReadFragA fetchA;
            load_matrix_coop_sync(fetchA, baseA, lda, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetA() + waveOffsetA(),
                                   reinterpret_cast<LocalWriteFragA&>(fetchA),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
        }

        __device__ static inline void
            prefetchGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = MappingUtil::workgroupDim();
            auto           waveCoord    = MappingUtil::waveCoord();
            auto           waveIndex    = std::get<0>(waveCoord);
            auto           waveCount    = std::get<0>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Issue global read
            GlobalReadFragB fetchB;
            load_matrix_coop_sync(fetchB, baseB, ldb, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetB() + waveOffsetB(),
                                   reinterpret_cast<LocalWriteFragB&>(fetchB),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
        }

        __device__ static inline void
            prefetchLocalA(FragA& fragA, DataT const* baseLds, uint32_t blockX)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragA&>(fragA),
                             baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(blockX),
                             ld());
        }

        __device__ static inline void
            prefetchLocalB(FragB& fragB, DataT const* baseLds, uint32_t blockY)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragB&>(fragB),
                             baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(blockY),
                             ld());
        }

        __device__ static inline void prefetchLocalA(FragA* fragsA, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                prefetchLocalA(fragsA[i], baseLds, i);
            }
        }

        __device__ static inline void prefetchLocalB(FragB* fragsB, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                prefetchLocalB(fragsB[i], baseLds, i);
            }
        }
    };

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil<BlockM,
                          BlockN,
                          BlockK,
                          DataT,
                          LayoutA,
                          LayoutB,
                          LayoutLds,
                          LdsKH,
                          BlocksX,
                          BlocksY>
    {
        /* LdsKH (BlockK = LDS Height)
    * Matrix geometry for A and B have a common dimension (BlockK).
    * We can fix one of the LDS dimensions to BlockK (in this case the height),
    * and insert blocks of different widths (BlockM, BlockN) to use the space
    * without the need of extra padding.
    *
    * This format is naturally conducive to matrix_b layout, whereas matrix_a
    * blocks are transposed.
    *
    * In this case, block dimensions for A and B are extended by
    * respective BlocksX and BlocksY, this allows for more efficient global
    * loads. These dimensions are also preserved in LDS, in which smaller
    * BlockMNK can be easily indexed.
    *
    * Local Layout (LDS):
    *
    *  kDim                    A (T)                              B
    *  |       -->   ______________  ...  ___           ______________  ...  ___
    *  |       |    |__________C0__  ...  ___|   ...   |__________R0__  ...  ___|
    *  |       |    |__________C1__  ...  ___|   ...   |__________R1__  ...  ___|
    *  | BlockK|    |__________C2__  ...  ___|   ...   |__________R2__  ...  ___|
    *  |       |    |          ...   ...     |   ...   |          ...   ...     |
    *  |       |    |__________Ck__  ...  ___|   ...   |__________Rk__  ...  ___|
    *  v        -->
    *               ^--- BlockM * BlocksX ----^         ^--- BlockN * BlocksY ---^
    */

        // Global read - extended leading dimension across all blocks
        // A matrix extends across BlocksX
        // B matrix extends across BlocksY
        using GlobalReadFragA
            = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, DataT, LayoutA>;
        using GlobalReadFragB
            = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, DataT, LayoutB>;

        // Local Write - extended leading dimension across all blocks
        // A matrix is transposed
        // B matrix already aligned
        using LocalWriteFragA
            = fragment<matrix_b, BlockN, BlockM * BlocksX, BlockK, DataT, LayoutLds>;
        using LocalWriteFragB
            = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, DataT, LayoutLds>;

        // Local Read - smaller block size target loads
        // A matrix reads BlockM x BlockK (transposed)
        // B matrix reads BlockK x BlockN
        using LocalReadFragA = fragment<matrix_b, BlockN, BlockM, BlockK, DataT, LayoutLds>;
        using LocalReadFragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutLds>;

        // General purpose fragment type
        using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
        using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        using MappingUtil = MappingUtil<BlockM, BlockN, DataT, LayoutLds>;

        __device__ static inline auto ldsWidth()
        {
            auto workgroupDim = MappingUtil::workgroupDim();
            return BlockM * BlocksX * std::get<0>(workgroupDim)
                   + BlockN * BlocksY * std::get<1>(workgroupDim);
        }

        __device__ static inline auto ldsHeight()
        {
            return BlockK;
        }

        __device__ static inline uint32_t ld()
        {
            return std::is_same<LayoutLds, row_major>::value ? ldsWidth() : ldsHeight();
        }

        __device__ static inline uint32_t baseOffsetA()
        {
            return 0;
        }

        __device__ static inline uint32_t baseOffsetB()
        {
            auto matrixCoord
                = std::make_pair(0, BlockM * BlocksX * std::get<0>(MappingUtil::workgroupDim()));
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetA()
        {
            auto matrixCoord
                = std::make_pair(0, BlockM * BlocksX * std::get<0>(MappingUtil::waveCoord()));
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetB()
        {
            auto matrixCoord
                = std::make_pair(0, BlockN * BlocksY * std::get<1>(MappingUtil::waveCoord()));
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
        {
            auto matrixCoord = std::make_pair(0, BlockM * blockX);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
        {
            auto matrixCoord = std::make_pair(0, BlockN * blockY);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline void
            prefetchGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = MappingUtil::workgroupDim();
            auto           waveCoord    = MappingUtil::waveCoord();
            auto           waveIndex    = std::get<1>(waveCoord);
            auto           waveCount    = std::get<1>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Share the load across all waves, we must do so for each row
            //for(int32_t i = 0; i < std::get<0>(workgroupDim); ++i)
            //{
            GlobalReadFragA fetchA;
            load_matrix_coop_sync(fetchA, baseA, lda, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetA() + waveOffsetA(),
                                   reinterpret_cast<LocalWriteFragA&>(fetchA),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
            //}
        }

        __device__ static inline void
            prefetchGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto           workgroupDim = MappingUtil::workgroupDim();
            auto           waveCoord    = MappingUtil::waveCoord();
            auto           waveIndex    = std::get<0>(waveCoord);
            auto           waveCount    = std::get<0>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Issue global read
            GlobalReadFragB fetchB;
            load_matrix_coop_sync(fetchB, baseB, ldb, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetB() + waveOffsetB(),
                                   reinterpret_cast<LocalWriteFragB&>(fetchB),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
        }

        __device__ static inline void
            prefetchLocalA(FragA& fragA, DataT const* baseLds, uint32_t blockX)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragA&>(fragA),
                             baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(blockX),
                             ld());
        }

        __device__ static inline void
            prefetchLocalB(FragB& fragB, DataT const* baseLds, uint32_t blockY)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragB&>(fragB),
                             baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(blockY),
                             ld());
        }

        __device__ static inline void prefetchLocalA(FragA* fragsA, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                prefetchLocalA(fragsA[i], baseLds, i);
            }
        }

        __device__ static inline void prefetchLocalB(FragB* fragsB, DataT const* baseLds)
        {
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                prefetchLocalB(fragsB[i], baseLds, i);
            }
        }
    };
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              uint32_t BlockSpanX,
              uint32_t BlockSpanY>
    struct LdsMappingUtil<BlockM,
                          BlockN,
                          BlockK,
                          DataT,
                          LayoutA,
                          LayoutB,
                          LayoutLds,
                          LdsKHC,
                          BlockSpanX,
                          BlockSpanY>
    {
        /* LdsKH (BlockK = LDS Height)
    * Matrix geometry for A and B have a common dimension (BlockK).
    * We can fix one of the LDS dimensions to BlockK (in this case the height),
    * and insert blocks of different widths (BlockM, BlockN) to use the space
    * without the need of extra padding.
    *
    * This format is naturally conducive to matrix_b layout, whereas matrix_a
    * blocks are transposed.
    *
    * In this case, block dimensions for A and B are extended by
    * respective BlocksX and BlocksY, this allows for more efficient global
    * loads. These dimensions are also preserved in LDS, in which smaller
    * BlockMNK can be easily indexed.
    *
    * Local Layout (LDS):
    *
    *  kDim                    A (T)                              B
    *  |       -->   ______________  ...  ___           ______________  ...  ___
    *  |       |    |__________C0__  ...  ___|   ...   |__________R0__  ...  ___|
    *  |       |    |__________C1__  ...  ___|   ...   |__________R1__  ...  ___|
    *  | BlockK|    |__________C2__  ...  ___|   ...   |__________R2__  ...  ___|
    *  |       |    |          ...   ...     |   ...   |          ...   ...     |
    *  |       |    |__________Ck__  ...  ___|   ...   |__________Rk__  ...  ___|
    *  v        -->
    *               ^--- BlockM * BlocksX ----^         ^--- BlockN * BlocksY ---^
    */

        // Global read - extended leading dimension across all blocks
        // A matrix extends across BlocksX
        // B matrix extends across BlocksY
        using GlobalReadFragA
            = fragment<matrix_a, BlockM * BlockSpanX, BlockN, BlockK, DataT, LayoutA>;
        using GlobalReadFragB
            = fragment<matrix_b, BlockM, BlockN * BlockSpanY, BlockK, DataT, LayoutB>;

        // Local Write - extended leading dimension across all blocks
        // A matrix is transposed
        // B matrix already aligned
        using LocalWriteFragA
            = fragment<matrix_b, BlockN, BlockM * BlockSpanX, BlockK, DataT, LayoutLds>;
        using LocalWriteFragB
            = fragment<matrix_b, BlockM, BlockN * BlockSpanY, BlockK, DataT, LayoutLds>;

        // Local Read - smaller block size target loads
        // A matrix reads BlockM x BlockK (transposed)
        // B matrix reads BlockK x BlockN
        using LocalReadFragA = fragment<matrix_b, BlockN, BlockM, BlockK, DataT, LayoutLds>;
        using LocalReadFragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutLds>;

        // General purpose fragment type
        using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
        using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        using MappingUtil = MappingUtil<BlockM, BlockN, DataT, LayoutLds>;

        __device__ static inline auto ldsWidth()
        {
            return BlockM * BlockSpanX + BlockN * BlockSpanY;
        }

        __device__ static inline auto ldsHeight()
        {
            return BlockK;
        }

        __device__ static inline uint32_t ld()
        {
            return std::is_same<LayoutLds, row_major>::value ? ldsWidth() : ldsHeight();
        }

        __device__ static inline uint32_t baseOffsetA()
        {
            return 0;
        }

        __device__ static inline uint32_t baseOffsetB()
        {
            auto matrixCoord = std::make_pair(0, BlockM * BlockSpanX);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
        {
            auto matrixCoord = std::make_pair(0, BlockM * blockX);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
        {
            auto matrixCoord = std::make_pair(0, BlockN * blockY);
            return MappingUtil::dataOffset(matrixCoord, ld());
        }

        __device__ static inline void
            prefetchGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto workgroupDim = MappingUtil::workgroupDim();
            auto waveCoord    = MappingUtil::waveCoord();
            auto waveIndex
                = std::get<0>(waveCoord) + std::get<1>(waveCoord) * std::get<0>(workgroupDim);
            // auto           waveCount    = std::get<0>(workgroupDim) * std::get<1>(workgroupDim);
            //auto           waveIndex    = std::get<0>(waveCoord) * std::get<1>(workgroupDim) + std::get<1>(waveCoord);
            auto           waveCount = std::get<0>(workgroupDim) * std::get<1>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragA::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragA::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Share the load across all waves, we must do so for each row
            GlobalReadFragA fetchA;
            load_matrix_coop_sync(fetchA, baseA, lda, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetA(),
                                   reinterpret_cast<LocalWriteFragA&>(fetchA),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
        }

        __device__ static inline void
            prefetchGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            // Tricky part: because we may be changing layouts and vector widths,
            // we need to ensure that splitCounts are the same on both sides of
            // global fetch and local writes - Otherwise the waves don't have the
            // same data responsibility.
            auto workgroupDim = MappingUtil::workgroupDim();
            auto waveCoord    = MappingUtil::waveCoord();
            auto waveIndex
                = std::get<0>(waveCoord) + std::get<1>(waveCoord) * std::get<0>(workgroupDim);
            //auto           waveIndex    = std::get<0>(waveCoord) * std::get<1>(workgroupDim) + std::get<1>(waveCoord);
            auto           waveCount = std::get<0>(workgroupDim) * std::get<1>(workgroupDim);
            constexpr auto splitCount
                = std::min((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount,
                           (uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount);

            static_assert(
                ((uint32_t)GlobalReadFragB::IOConfig::IOTraits::IOCount % splitCount == 0)
                    && ((uint32_t)LocalWriteFragB::IOConfig::IOTraits::IOCount % splitCount == 0),
                "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            // Issue global read
            GlobalReadFragB fetchB;
            load_matrix_coop_sync(fetchB, baseB, ldb, waveIndex, waveCount, splitCount);

            // Issue local store
            store_matrix_coop_sync(baseLds + baseOffsetB(),
                                   reinterpret_cast<LocalWriteFragB&>(fetchB),
                                   ld(),
                                   waveIndex,
                                   waveCount,
                                   splitCount);
        }

        __device__ static inline void
            prefetchLocalA(FragA& fragA, DataT const* baseLds, uint32_t blockX)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragA&>(fragA),
                             baseLds + baseOffsetA() + blockOffsetA(blockX),
                             ld());
        }

        __device__ static inline void
            prefetchLocalB(FragB& fragB, DataT const* baseLds, uint32_t blockY)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragB&>(fragB),
                             baseLds + baseOffsetB() + blockOffsetB(blockY),
                             ld());
        }
    };

} // namespace rocwmma

#endif // LDS_MAPPING_UTIL_H
