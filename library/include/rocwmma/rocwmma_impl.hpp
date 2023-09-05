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
#ifndef ROCWMMA_API_IMPL_HPP
#define ROCWMMA_API_IMPL_HPP

#include "rocwmma.hpp"

#include "internal/accessors.hpp"
#include "internal/blend.hpp"
#include "internal/broadcast.hpp"
#include "internal/constants.hpp"
#include "internal/convert.hpp"
#include "internal/dpp.hpp"
#include "internal/flow_control.hpp"
#include "internal/io_config.hpp"
#include "internal/io_shape.hpp"
#include "internal/io_traits.hpp"
#include "internal/layout.hpp"
#include "internal/mapping_util.hpp"
#include "internal/mfma.hpp"
#include "internal/opaque_load.hpp"
#include "internal/opaque_store.hpp"
#include "internal/pack_util.hpp"
#include "internal/permute.hpp"
#include "internal/swizzle.hpp"
#include "internal/transforms.hpp"
#include "internal/types.hpp"
#include "internal/utils.hpp"
#include "internal/vector.hpp"
#include "internal/vector_iterator.hpp"
#include "internal/wmma.hpp"

namespace rocwmma
{
    namespace detail
    {
        // Ensure that MFMA fragments for A and B have orthogonal layouts
        template <typename FragA, typename FragB>
        struct MfmaCheck : public MatrixLayout::detail::OrthogonalCheck<
                               typename GetIOShape_t<FragA>::MatrixLayout,
                               typename GetIOShape_t<FragB>::MatrixLayout>
        {
        };
    }

    // fragment implementations
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::fragment(const fragment& other)
        : mStorage(other.mStorage)
    {
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator=(
            const fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>& other)
    {
        mStorage = other.mStorage;
        return *this;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE inline DataT&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator[](uint32_t index)
    {
        return mAccess.data[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE inline auto
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator*() ->
        typename Traits::StorageT&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE inline DataT const&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator[](uint32_t index) const
    {
        return mAccess.data[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE inline auto
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::operator*() const ->
        typename Traits::StorageT const&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::blockDim()
    {
        return IOConfig::BlockDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::kDim()
    {
        return IOConfig::KDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, LayoutT>::size()
    {
        return num_elements;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                      DataT                                                         value)
    {
        using FragT       = typename std::decay<decltype(frag)>::type;
        using Broadcaster = typename GetIOConfig_t<FragT>::Broadcaster;

        // Sanity check
        static_assert(std::is_same<typename Broadcaster::Traits::BroadcastT,
                                   typename FragT::Traits::AccessT>::value,
                      "Broadcast input and fragment access types do not match");

        Broadcaster::exec(frag.mAccess, value);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                                  data,
                         uint32_t                                                      ldm)
    {
        using FragT  = typename std::decay<decltype(frag)>::type;
        using Loader = typename GetIOConfig_t<FragT>::Loader;

        // Sanity checks
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(
            std::is_same<typename FragT::Traits::AccessT, typename Loader::Traits::OutputT>::value,
            "Fragment access and load output types do not match");

        // Load then implicit pack
        Loader::exec(frag.mAccess, data, ldm);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                         const DataT*                                      data,
                                         uint32_t                                          ldm,
                                         layout_t                                          layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            load_matrix_sync(reinterpret_cast<FragRowMajor&>(frag), data, ldm);
        }
        else
        {
            load_matrix_sync(reinterpret_cast<FragColMajor&>(frag), data, ldm);
        }
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                              data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                            ldm)
    {
        using FragT  = typename std::decay<decltype(frag)>::type;
        using Storer = typename GetIOConfig_t<FragT>::Storer;

        // Sanity check
        static_assert(!std::is_same<DataLayout, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(
            std::is_same<typename FragT::Traits::AccessT, typename Storer::Traits::InputT>::value,
            "Fragment access and store input types do not match");

        // Implicit unpack and then store
        Storer::exec(data, frag.mAccess, ldm);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                  data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                          uint32_t                                                ldm,
                          layout_t                                                layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            store_matrix_sync(data, reinterpret_cast<FragRowMajor const&>(frag), ldm);
        }
        else
        {
            store_matrix_sync(data, reinterpret_cast<FragColMajor const&>(frag), ldm);
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    ROCWMMA_DEVICE void
        mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutD>&       d,
                 fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const&      a,
                 fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const&      b,
                 fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutC> const& c)
    {
        using FragA = typename std::decay<decltype(a)>::type;
        using FragB = typename std::decay<decltype(b)>::type;

        // Sanity check
        // static_assert(detail::MfmaCheck<FragA, FragB>::value,
        //              "A and B fragment layouts must be orthogonal");
        using MMA = typename std::conditional_t<ROCWMMA_ARCH_GFX9,
                                                Mfma<InputT, ComputeT, BlockM, BlockN, BlockK>,
                                                Wmma<InputT, ComputeT, BlockM, BlockN, BlockK>>;

        (*d) = MMA::exec(*a, *b, *c);
    }

    ROCWMMA_DEVICE void synchronize_workgroup()
    {
        __syncthreads();
    }

} // namespace rocwmma

#endif // ROCWMMA_API_IMPL_HPP
