/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_TRANSFORMS_API_IMPL_HPP
#define ROCWMMA_TRANSFORMS_API_IMPL_HPP

#include "internal/transforms.hpp"
#include "rocwmma_transforms.hpp"

namespace rocwmma
{
    namespace detail
    {
        ///
        /// Apply logical transpose of fragment
        ///

        // Below are defined as fast implicit transposes:
        // - We reinterpret meaning between cols of A and rows of B,
        // in order to change the shape of our data for reading / writing.
        // Implicit transposes of fragment objects are designed to be
        // relatively cheap, and should only require a signature cast.
        // Assumptions:
        // - BlockDim and KDim are identical
        // - Matrix Layouts are orthogonal (exchange rows / cols)
        // - Data layouts are orthogonal (exchange row / col major)
        // - Register layouts match. (No change)
        // Example:
        // - A matrix_a fragment of (BlockM x BlockK) = 32x8 in col_major may be reinterpreted
        //   as a matrix_b fragment of (BlockK x BlockN) = 8x32 in row_major.
        //   Here, we have transposed (reimagined) 8 cols of matrix_a into 8 rows of matrix_b.
        template <typename FragT>
        struct ApplyTranspose;

        template <uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT>
        struct ApplyTranspose<fragment<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
        {
        private:
            // Original frag A type
            using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT>;

            // Transpose to frag B type in opposite data layout.
            using FragB = fragment<matrix_b,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   orthogonal_layout_t<DataLayoutT>>;

            using IOConfigA = GetIOConfig_t<FragA>;
            using IOConfigB = GetIOConfig_t<FragB>;

            // Assumptions check
            static_assert(IOConfigA::IOShape::BlockDim == IOConfigB::IOShape::BlockDim,
                          "BlockDim of transposed frag doesn't match");

            static_assert(IOConfigA::IOShape::KDim == IOConfigB::IOShape::KDim,
                          "KDim of transposed fragm doesn't match");

            static_assert(is_orthogonal_v<typename IOConfigA::IOLayout::DataLayout,
                                          typename IOConfigB::IOLayout::DataLayout>,
                          "Data Layouts are not orthogonal");

            static_assert(is_orthogonal_v<typename IOConfigA::IOLayout::MatrixLayout,
                                          typename IOConfigB::IOLayout::MatrixLayout>,
                          "Matrix Layouts are not orthogonal");

            static_assert(std::is_same_v<typename IOConfigA::IOLayout::RegisterLayout,
                                         typename IOConfigB::IOLayout::RegisterLayout>,
                          "Register layouts do not match");

        public:
            // Interface
            using Type = FragB;

            // Because of the expectation that matrix_a data is orthogonal to matrix_b
            // with the same register layout, the transpose comes as a simple re-cast.
            ROCWMMA_DEVICE static inline FragB const& exec(FragA const& frag)
            {
                return reinterpret_cast<FragB const&>(frag);
            }
        };

        template <uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT>
        struct ApplyTranspose<fragment<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
        {
        private:
            // Original frag A type
            using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT>;

            // Transpose to frag A type in opposite data layout.
            using FragA = fragment<matrix_a,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   orthogonal_layout_t<DataLayoutT>>;

            using IOConfigA = GetIOConfig_t<FragA>;
            using IOConfigB = GetIOConfig_t<FragB>;

            // Assumptions check
            static_assert(IOConfigA::IOShape::BlockDim == IOConfigB::IOShape::BlockDim,
                          "BlockDim of transposed frag doesn't match");

            static_assert(IOConfigA::IOShape::KDim == IOConfigB::IOShape::KDim,
                          "KDim of transposed frag doesn't match");

            static_assert(is_orthogonal_v<typename IOConfigA::IOLayout::DataLayout,
                                          typename IOConfigB::IOLayout::DataLayout>,
                          "Data Layouts are not orthogonal");

            static_assert(is_orthogonal_v<typename IOConfigA::IOLayout::MatrixLayout,
                                          typename IOConfigB::IOLayout::MatrixLayout>,
                          "Matrix Layouts are not orthogonal");

            static_assert(std::is_same_v<typename IOConfigA::IOLayout::RegisterLayout,
                                         typename IOConfigB::IOLayout::RegisterLayout>,
                          "Register layouts do not match");

        public:
            // Interface
            using Type = FragA;

            // Because of the expectation that matrix_a data is orthogonal to matrix_b
            // with the same register layout, the transpose comes as a simple re-cast.
            ROCWMMA_DEVICE static inline FragA const& exec(FragB const& frag)
            {
                return reinterpret_cast<FragA const&>(frag);
            }
        };

        // Below are defined data layout transforms:
        // - The same fragment data is to be re-arranged in the format
        //   of another another Data Layout.
        // - This can be achieved through AOS<->SOA transformations
        //   where required, as long as the matrix context, block
        //   dimensions and MaxVectorWidths have not changed.
        // Assumptions:
        // - Matrix contexts are identical, as well as Block dimensions
        // - Matrix layouts may change, but adhere to strict AOS or SOA formats.
        // - Register layout transforms are needed when they do not match
        // Example:
        // - A matrix_a fragment of (BlockM x BlockK) = 32x8 in col_major may be reinterpreted as
        //   a matrix_a fragment of (BlockM x BlockK) = 32x8 in row_major.
        //   Here, we have rearranged col_major for row_major ordering.
        template <typename FragT, typename NewDataLayoutT>
        struct ApplyDataLayout;

        // Same layout case
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT>
        struct ApplyDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>,
                               DataLayoutT>
        {
            // Interface
            using Type = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>;
            template <uint32_t WaveCount = 1>
            ROCWMMA_DEVICE constexpr static inline Type const& exec(Type const& frag)
            {
                return frag;
            }
        };

        // Other layout case
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  typename NewDataLayoutT>
        struct ApplyDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>,
                               NewDataLayoutT>
        {
        private:
            using FragIn  = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>;
            using FragOut = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, NewDataLayoutT>;

            using IOConfigIn = GetIOConfig_t<FragIn>;

            using RegisterLayoutIn  = typename GetIOConfig_t<FragIn>::IOLayout::RegisterLayout;
            using RegisterLayoutOut = typename GetIOConfig_t<FragOut>::IOLayout::RegisterLayout;

            // Matrix context, BlockDim and KDim implicitly the same due to re-use of
            // MatrixT, BlockM, BlockN, BlockK

        public:
            // Interface
            using Type = FragOut;

            // Optimal case: input and output register layouts match
            template <uint32_t WaveCount = 1,
                      typename FragT,
                      typename std::enable_if_t<
                          std::is_same_v<FragT, FragIn>
                              && std::is_same_v<RegisterLayoutIn, RegisterLayoutOut>,
                          int>
                      = 0>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(FragT const& frag)
            {
                return reinterpret_cast<FragOut const&>(frag);
            }

            // Input and output register layouts do not match: must transform using AOS<->SOA
            template <uint32_t WaveCount = 1,
                      typename FragT,
                      typename std::enable_if_t<
                          std::is_same_v<FragT, FragIn>
                              && !std::is_same_v<RegisterLayoutIn, RegisterLayoutOut>,
                          int>
                      = 0>
            ROCWMMA_DEVICE constexpr static inline auto exec(FragT const& frag)
            {
                // TODO: Make sure to use coop configs to get the right MaxVW!!!
                using IOConfigCoop           = GetCoopIOConfig_t<FragIn, WaveCount>;
                constexpr uint32_t BlockDim  = IOConfigCoop::IOShape::BlockDim;
                constexpr uint32_t MaxVW     = IOConfigCoop::IOLayout::MaxVW;
                using RegisterLayoutIncoming = typename IOConfigCoop::IOLayout::RegisterLayout;

                // Target layouts
                using AosLayout = RegisterLayout::template Aos<BlockDim, MaxVW>;
                using SoaLayout = RegisterLayout::template Soa<BlockDim, MaxVW>;

                auto result = FragOut{};

                if constexpr(std::is_same_v<AosLayout, RegisterLayoutIncoming>)
                {
                    result.mAccess = IAosToSoa<BlockDim, MaxVW>::exec(frag.mAccess);
                }
                else if constexpr(std::is_same_v<SoaLayout, RegisterLayoutIncoming>)
                {
                    result.mAccess = ISoaToAos<BlockDim, MaxVW>::exec(frag.mAccess);
                }

                return result;
            }
        };

        template <typename FragT>
        struct ApplyRegisterFile;
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout>
        struct ApplyRegisterFile<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>>
        {
        private:
            using FragT = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;
            constexpr static const uint32_t registerFileWidth = Constants::AMDGCN_WAVE_SIZE;

        public:
            using Type = fragment<matrix_b, 1, registerFileWidth, FragT::size(), DataT, DataLayout>;
        };

    } // namespace detail

    /// These wrappers must perfect-forward and perfect-return because the return types and
    // arguments above could be references or copy types.
    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) applyTranspose(FragT&& frag)
    {
        return detail::template ApplyTranspose<std::decay_t<FragT>>::exec(
            std::forward<FragT>(frag));
    }

    template <typename DataLayoutT, uint32_t WaveCount /*=1*/, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) applyDataLayout(FragT&& frag)
    {
        return detail::template ApplyDataLayout<std::decay_t<FragT>, DataLayoutT>::template exec<
            WaveCount>(std::forward<FragT>(frag));
    }

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_API_IMPL_HPP
