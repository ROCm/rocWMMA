/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

namespace rocwmma
{
    namespace detail
    {
        ///
        /// Consistency and orthogonality checks as they apply to fragment types
        ///
        template <typename LhsFrag, typename RhsFrag>
        struct ConsistencyCheck : public MatrixLayout::detail::ConsistencyCheck<
                                      typename GetIOShape_t<LhsFrag>::MatrixLayout,
                                      typename GetIOShape_t<RhsFrag>::MatrixLayout>
        {
        };

        template <typename LhsFrag, typename RhsFrag>
        struct OrthogonalCheck : public MatrixLayout::detail::OrthogonalCheck<
                                     typename GetIOShape_t<LhsFrag>::MatrixLayout,
                                     typename GetIOShape_t<RhsFrag>::MatrixLayout>
        {
        };

        ///
        /// Apply implicit transpose of fragment
        ///
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
            // Original frag type
            using Frag = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT>;

            // Transposed frag type
            using FragT = fragment<matrix_b,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   typename DataLayout::template OrthogonalLayout_t<DataLayoutT>>;

            // Sanity check
            static_assert(OrthogonalCheck<Frag, FragT>::value,
                          "Implicit fragment transpose is not orthogonal");

        public:
            using Type = FragT;

            ROCWMMA_DEVICE static inline Type const& exec(Frag const& frag)
            {
                return reinterpret_cast<Type const&>(frag);
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
            // Original frag type
            using Frag = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT>;

            // Transposed frag type
            using FragT = fragment<matrix_a,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   typename DataLayout::template OrthogonalLayout_t<DataLayoutT>>;

            // Sanity check
            static_assert(OrthogonalCheck<Frag, FragT>::value,
                          "Implicit fragment transpose failed");

        public:
            using Type = FragT;

            ROCWMMA_DEVICE static inline Type const& exec(Frag const& frag)
            {
                return reinterpret_cast<Type const&>(frag);
            }
        };

        ///
        /// Apply implicit data layout change of fragment
        ///
        template <typename FragT, typename NewDataLayoutT>
        struct ApplyDataLayout;

        // Same layout case
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout>
        struct ApplyDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>,
                               DataLayout>
        {
            using Type = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;
            ROCWMMA_DEVICE constexpr static inline Type const& exec(Type const& frag)
            {
                return frag;
            }
        };

        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayout,
                  typename NewDataLayout>
        struct ApplyDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>,
                               NewDataLayout>
        {
        private:
            using Frag  = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;
            using FragT = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, NewDataLayout>;

            // Some fragment layouts like ColNT and RowNT enforce consistency across DataLayouts.
            // If so, we can implicitly change the DataLayout.
            static_assert(ConsistencyCheck<Frag, FragT>::value,
                          "Implicit fragment DataLayout change is inconsistent");

        public:
            using Type = FragT;

            ROCWMMA_DEVICE constexpr static inline Type const& exec(Frag const& frag)
            {
                return reinterpret_cast<Type const&>(frag);
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

    template <typename DataLayoutT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) applyDataLayout(FragT&& frag)
    {
        return detail::template ApplyDataLayout<std::decay_t<FragT>, DataLayoutT>::exec(
            std::forward<FragT>(frag));
    }

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_API_IMPL_HPP
