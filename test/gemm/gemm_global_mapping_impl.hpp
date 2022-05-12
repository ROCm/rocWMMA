/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef GEMM_GLOBAL_MAPPING_IMPL_HPP
#define GEMM_GLOBAL_MAPPING_IMPL_HPP

#include "gemm_global_mapping.hpp"

namespace rocwmma
{
    namespace CooperativeGemm
    {

#define GlobalMappingT                                                                             \
    uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename InputT, typename OutputT,          \
        typename ComputeT, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD, \
        uint32_t BlocksX, uint32_t BlocksY

#define GlobalMappingT_impl                                                                \
    BlockM, BlockN, BlockK, InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD, \
        BlocksX, BlocksY

        template <GlobalMappingT>
        template <typename CoordC>
        __device__ constexpr inline auto
            GlobalMapping<GlobalMappingT_impl>::projCoordA(CoordC const& coordC)
        {
            return std::make_pair(std::get<0>(coordC), 0u);
        }

        template <GlobalMappingT>
        template <typename CoordC>
        __device__ constexpr inline auto
            GlobalMapping<GlobalMappingT_impl>::projCoordB(CoordC const& coordC)
        {
            return std::make_pair(0u, std::get<1>(coordC));
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::macroTileSizeC()
        {
            return detail::WaveSpace::workgroupDim()
                   * std::make_pair((uint32_t)IOShape<GRFragA>::BlockHeight,
                                    (uint32_t)IOShape<GRFragB>::BlockWidth);
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::macroTileCoordC()
        {
            return detail::WaveSpace::workgroupCoord() * macroTileSizeC();
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::waveOffsetA()
        {
            return projCoordA(waveOffsetC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::waveOffsetB()
        {
            return projCoordB(waveOffsetC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::waveOffsetC()
        {
            return detail::WaveSpace::localWaveCoord()
                   * std::make_pair((uint32_t)IOShape<GRFragA>::BlockHeight,
                                    (uint32_t)IOShape<GRFragB>::BlockWidth);
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::blockOffsetA()
        {
            return projCoordA(blockOffsetC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::blockOffsetB()
        {
            return projCoordB(blockOffsetC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::blockOffsetC()
        {
            return std::make_pair((uint32_t)IOShape<MfmaFragA>::BlockHeight,
                                  (uint32_t)IOShape<MfmaFragB>::BlockWidth);
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::matrixCoordA()
        {
            return projCoordA(matrixCoordC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::matrixCoordB()
        {
            return projCoordB(matrixCoordC());
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::matrixCoordC()
        {
            return macroTileCoordC() + waveOffsetC();
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::kStepA()
        {
            // Sanity check
            static_assert((uint32_t)IOShape<GRFragA>::BlockWidth == BlockK,
                          "Global read A block width does not match BlockK step dimension");

            return std::make_pair(0u, BlockK);
        }

        template <GlobalMappingT>
        __device__ constexpr inline auto GlobalMapping<GlobalMappingT_impl>::kStepB()
        {
            // Sanity check
            static_assert((uint32_t)IOShape<GRFragB>::BlockHeight == BlockK,
                          "Global read B block height does not match BlockK step dimension");

            return std::make_pair(BlockK, 0u);
        }

#undef GlobalMappingT
#undef GlobalMappingT_impl

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_GLOBAL_MAPPING_IMPL_HPP
