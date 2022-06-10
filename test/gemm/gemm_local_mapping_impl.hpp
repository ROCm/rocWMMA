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
#ifndef GEMM_LOCAL_MAPPING_IMPL_HPP
#define GEMM_LOCAL_MAPPING_IMPL_HPP

#include "gemm_global_mapping.hpp"
#include "gemm_local_mapping.hpp"

namespace rocwmma
{
    namespace LocalMapping
    {

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::waveOffsetA()
        {
            return std::swap(GlobalMapping::waveOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::waveOffsetB()
        {
            return GlobalMapping::waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::blockOffsetA()
        {
            return std::swap(GlobalMapping::blockOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::blockOffsetB()
        {
            return GlobalMapping::blockOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::matrixCoordA()
        {
            // Base lds coord = (0, 0);
            return waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::matrixCoordB()
        {
            return std::swap(GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC()))
                   + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return std::make_pair(LdsHeight, std::get<0>(macroTileC) + std::get<1>(macroTileC));
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::waveOffsetA()
        {
            return GlobalMapping::waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::waveOffsetB()
        {
            return std::swap(GlobalMapping::waveOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::blockOffsetA()
        {
            return GlobalMapping::blockOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::blockOffsetB()
        {
            return std::swap(GlobalMapping::blockOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::matrixCoordA()
        {
            // Assuming base coord of (0, 0) for LDS
            return waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::matrixCoordB()
        {
            return GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC()) + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return std::make_pair(std::get<0>(macroTileC) + std::get<1>(macroTileC), LdsWidth);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

#undef LdsMappingT
#undef LdsMappingT_impl

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto
            LdsMappingRF<LdsMappingT_impl>::projCoordA(Coord2d const& coordA)
        {
            // Scale the A coordinate to register file height
            return std::make_pair(std::get<0>(coordA) * GlobalMapping::kDim() / LdsWidth, 0u);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto
            LdsMappingRF<LdsMappingT_impl>::projCoordB(Coord2d const& coordB)
        {
            // Scale the B coordinate to register file height
            return std::make_pair(std::get<1>(coordB) * GlobalMapping::kDim() / LdsWidth, 0u);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::waveOffsetA()
        {
            return projCoordA(GlobalMapping::waveOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::waveOffsetB()
        {
            return projCoordB(GlobalMapping::waveOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::blockOffsetA()
        {
            return projCoordA(GlobalMapping::blockOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::blockOffsetB()
        {
            return projCoordB(GlobalMapping::blockOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::matrixCoordA()
        {
            // Assuming base coord of (0, 0) for LDS
            return waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::matrixCoordB()
        {
            return projCoordA(GlobalMapping::macroTileSizeC()) + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return std::make_pair((std::get<0>(macroTileC) + std::get<1>(macroTileC))
                                      * GlobalMapping::kDim() / LdsWidth,
                                  LdsWidth);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

    } // namespace LocalMapping

} // namespace rocwmma

#endif // GEMM_LOCAL_MAPPING_IMPL_HPP
