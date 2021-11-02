#ifndef LDS_MAPPING_UTIL_H
#define LDS_MAPPING_UTIL_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "WMMA.h"
#pragma GCC diagnostic pop

#include "MappingUtil.h"
#include "Utils.h"

class LdsKW
{
};
class LdsKH
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
    static constexpr uint32_t registerFileWidth = AMDGCN_WAVE_SIZE;

    using FragA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
    using FragB = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

    using MappingLdsA = MappingUtil<BlockM, BlockK, DataT, LayoutLds>;
    using MappingLdsB = MappingUtil<BlockN, BlockK, DataT, LayoutLds>;

    using FragLdsA = wmma::
        fragment<register_file_coop_a, 1, registerFileWidth, FragA::size(), DataT, LayoutLds>;
    using FragLdsB = wmma::
        fragment<register_file_coop_b, 1, registerFileWidth, FragB::size(), DataT, LayoutLds>;

    __device__ static inline auto ldsWidth()
    {
        return registerFileWidth;
    }

    __device__ static inline auto ldsHeight()
    {
        auto workgroupDim = MappingLdsA::workgroupDim();
        return FragLdsA::size() * BlocksX * std::get<0>(workgroupDim)
               + FragLdsB::size() * BlocksY * std::get<1>(workgroupDim);
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
            FragLdsA::size() * BlocksX * std::get<0>(MappingLdsA::workgroupDim()), 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetA()
    {
        auto matrixCoord
            = std::make_pair(FragLdsA::size() * BlocksX * std::get<0>(MappingLdsA::waveCoord()), 0);
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetB()
    {
        auto matrixCoord
            = std::make_pair(FragLdsB::size() * BlocksY * std::get<1>(MappingLdsB::waveCoord()), 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
    {
        auto matrixCoord = std::make_pair(FragLdsA::size() * blockX, 0);
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
    {
        auto matrixCoord = std::make_pair(FragLdsB::size() * blockY, 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
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
    using MappingLdsA = MappingUtil<BlockM, BlockK, DataT, LayoutLds>;
    using MappingLdsB = MappingUtil<BlockN, BlockK, DataT, LayoutLds>;

    using FragLdsA = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutLds>;
    using FragLdsB = wmma::fragment<matrix_b, BlockM, BlockK, BlockN, DataT, LayoutLds>;

    __device__ static inline auto ldsWidth()
    {
        return BlockK;
    }

    __device__ static inline auto ldsHeight()
    {
        auto workgroupDim = MappingLdsA::workgroupDim();
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
            = std::make_pair(BlockM * BlocksX * std::get<0>(MappingLdsA::workgroupDim()), 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetA()
    {
        auto matrixCoord
            = std::make_pair(BlockM * BlocksX * std::get<0>(MappingLdsA::waveCoord()), 0);
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetB()
    {
        auto matrixCoord
            = std::make_pair(BlockN * BlocksY * std::get<1>(MappingLdsB::waveCoord()), 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
    {
        auto matrixCoord = std::make_pair(BlockM * blockX, 0);
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
    {
        auto matrixCoord = std::make_pair(BlockN * blockY, 0);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
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
    using MappingLdsA = MappingUtil<BlockM, BlockK, DataT, LayoutLds>;
    using MappingLdsB = MappingUtil<BlockN, BlockK, DataT, LayoutLds>;

    using FragLdsA = wmma::fragment<matrix_a, BlockK, BlockN, BlockM, DataT, LayoutLds>;
    using FragLdsB = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutLds>;

    __device__ static inline auto ldsWidth()
    {
        auto workgroupDim = MappingLdsA::workgroupDim();
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
            = std::make_pair(0, BlockM * BlocksX * std::get<0>(MappingLdsA::workgroupDim()));
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetA()
    {
        auto matrixCoord
            = std::make_pair(0, BlockM * BlocksX * std::get<0>(MappingLdsA::waveCoord()));
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t waveOffsetB()
    {
        auto matrixCoord
            = std::make_pair(0, BlockN * BlocksY * std::get<1>(MappingLdsB::waveCoord()));
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
    {
        auto matrixCoord = std::make_pair(0, BlockM * blockX);
        return MappingLdsA::dataOffset(ld(), matrixCoord);
    }

    __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
    {
        auto matrixCoord = std::make_pair(0, BlockN * blockY);
        return MappingLdsB::dataOffset(ld(), matrixCoord);
    }
};

#endif // LDS_MAPPING_UTIL_H
