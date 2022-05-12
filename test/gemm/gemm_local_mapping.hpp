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
#ifndef GEMM_LOCAL_MAPPING_HPP
#define GEMM_LOCAL_MAPPING_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace CooperativeGemm
    {
        template <typename GlobalMapping, typename LayoutLds>
        struct LdsMappingTN
        {
            /* 
            * This is a helper class to transform global A and B fragments such that they can
            * fit together in a common LDS space without the use of extra padding.
            * We use the fragments defined in the GlobalMapping argument.
            * 
            * LdsMappingTN (Block Height = LDS Height)
            * Matrix geometry for A and B have a common dimension in BlockK.
            * We can fix the height of the LDS storage to BlockK, and transform
            * fragments of A and B to fit without extra padding.
            *
            * Fragments of A must be transposed to fit this geometry,
            * and both fragments from A and B must accomodate LDS data layout.
            *
            * Local Matrix Layout (LDS):
            * 
            * Transposed A fragments [A0 (T) ... AX-1 (T)] are placed first and occupy a total width of MacroTileX.
            * B fragments [B0 ... BY-1] follow A fragments and occupy a total width of MacroTileY
            * 
            *
            *  kDim                        A0 (T)             AX-1 (T)            B0                BY-1
            *  |      (0,0)  -->  ______________  ...  ___               ______________  ...  ___
            *  |             |   |__________C0__  ...  ___|     ...     |__________R0__  ...  ___|  ...  
            *  |             |   |__________C1__  ...  ___|     ...     |__________R1__  ...  ___|  ...
            *  |      BlockK |   |__________C2__  ...  ___|     ...     |__________R2__  ...  ___|  ...
            *  |             |   |          ...   ...     |     ...     |          ...   ...     |  ...
            *  |             |   |__________Ck__  ...  ___|     ...     |__________Rk__  ...  ___|  ...
            *  v             -->
            *                    ^------- BlockM ---------^             ^-------- BlockN --------^
            * 
            *                    ^-------- MacroTileX ----------^ ^----------- MacroTileY ------------^ (BlockK-1, MacroTileX + MacroTileY - 1)
            */

            // Helper to quickly access IOShape properties
            template<typename FragT>
            using IOShape = typename FragT::IOConfig::IOShape;

            using DataSpace = detail::DataSpace<LayoutLds>;

            /// LOCAL WRITE <- Collaborative frags
            // K = BlockHeight
            // GlobalReadFragA Transposed
            // GlobalReadFragB unchanged 
            // Ensure LW frags are LayoutLds friendly.
            using LWFragA = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::GRFragA>, LayoutLds>;
            using LWFragB = ApplyDataLayout_t<typename GlobalMapping::GRFragB, LayoutLds>;

            /// LOCAL READ -> Mfma frags
            // Reading transposed Mfma A blocks
            // Reading unchanged Mfma B blocks
            // Ensure LR frags are LayoutLds friendly.
            using LRFragA = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::MfmaFragA>, LayoutLds>;
            using LRFragB = ApplyDataLayout_t<typename GlobalMapping::MfmaFragB, LayoutLds>;

            /// Sanity checks:
            // All local R/W tiles should have same height
            static_assert(IOShape<LWFragA>::BlockHeight == IOShape<LWFragB>::BlockHeight, "LW frag heights do not match");
            static_assert(IOShape<LRFragA>::BlockHeight == IOShape<LRFragB>::BlockHeight, "LR frag heights do not match");
            static_assert(IOShape<LWFragA>::BlockHeight == IOShape<LRFragA>::BlockHeight, "LW and LR frag heights do not match");

            // Heights should equal KDim
            static_assert(IOShape<LWFragA>::KDim == IOShape<LWFragB>::KDim, "LW frag K dims do not match");
            static_assert(IOShape<LRFragA>::KDim == IOShape<LRFragB>::KDim, "LR frag K dims do not match");
            static_assert(IOShape<LWFragA>::KDim == IOShape<LRFragA>::KDim, "LW and LR frag K dims do not match");

            // Finally, height should equal KDim
            static_assert(IOShape<LWFragA>::BlockHeight == IOShape<LWFragB>::KDim, "Frag height does not equal K dim");

            constexpr static uint32_t LdsHeight = IOShape<LWFragA>::BlockHeight;

            // Offset of the current wave in the LDS macro tile
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();

            // Block offset between local mfma fragments
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();

            // The base lds matrix coordinate for current wave
            __device__ constexpr static inline auto matrixCoordA();
            __device__ constexpr static inline auto matrixCoordB();

            // Dimensions of shared memory usage
            __device__ constexpr static inline auto sizeLds();

            // Leading dimension of lds matrix
            __device__ constexpr static inline auto ldLds();
        };

        template <typename GlobalMapping, typename LayoutLds>
        struct LdsMappingNT
        {
            /* LdsMappingNT (BlockK = LDS Width)
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

            // Helper to quickly access IOShape properties
            template<typename FragT>
            using IOShape = typename FragT::IOConfig::IOShape;

            using DataSpace = detail::DataSpace<LayoutLds>;

            /// LOCAL WRITE
            // K = BlockWidth
            // GlobalReadFragA unchanged
            // GlobalReadFragB Transposed 
            // Ensure LW frags are LayoutLds friendly.
            using LWFragA = ApplyDataLayout_t<typename GlobalMapping::GRFragA, LayoutLds>;
            using LWFragB = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::GRFragB>, LayoutLds>;

            /// LOCAL READ
            // Reading unchanged Mfma A blocks
            // Reading transposed Mfma B blocks
            // Ensure LR frags are LayoutLds friendly.
            using LRFragA = ApplyDataLayout_t<typename GlobalMapping::MfmaFragA, LayoutLds>;
            using LRFragB = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::MfmaFragB>, LayoutLds>;

            // Sanity check:
            // All local R/W tiles should have same width
            static_assert(IOShape<LWFragA>::BlockWidth == IOShape<LWFragB>::BlockWidth, "LW frag widths do not match");
            static_assert(IOShape<LRFragA>::BlockWidth == IOShape<LRFragB>::BlockWidth, "LR frag widths do not match");
            static_assert(IOShape<LWFragA>::BlockWidth == IOShape<LRFragA>::BlockWidth, "LW and LR frag widths do not match");

            // Heights should equal KDim
            static_assert(IOShape<LWFragA>::KDim == IOShape<LWFragB>::KDim, "LW frag K dims do not match");
            static_assert(IOShape<LRFragA>::KDim == IOShape<LRFragB>::KDim, "LR frag K dims do not match");
            static_assert(IOShape<LWFragA>::KDim == IOShape<LRFragA>::KDim, "LW and LR frag K dims do not match");

            // Finally, height should equal KDim
            static_assert(IOShape<LWFragA>::BlockWidth == IOShape<LWFragB>::KDim, "Frag width does not equal K dim");

            constexpr static uint32_t LdsWidth = IOShape<LWFragA>::BlockWidth;

            // Offset of the current wave in the LDS macro tile
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();

            // Block offset between local mfma fragments
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();

            // The base lds matrix coordinate for current wave
            __device__ constexpr static inline auto matrixCoordA();
            __device__ constexpr static inline auto matrixCoordB();

            // Dimensions of shared memory usage
            __device__ constexpr static inline auto sizeLds();

            // Leading dimension of shared memory usage
            __device__ constexpr static inline auto ldLds();
        };

    } // namespace CooperativeGemm

} // namespace rocwmma

#include "gemm_local_mapping_impl.hpp"

#endif // GEMM_LOCAL_MAPPING_HPP