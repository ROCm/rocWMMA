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
#ifndef ROCWMMA_API_HPP
#define ROCWMMA_API_HPP

#include "internal/accessors.hpp"
#include "internal/io_traits.hpp"
#include "internal/pack_util.hpp"
#include "internal/types.hpp"

/**
 * \mainpage
 *
 * rocWMMA is a C++ header library for accelerating mixed precision matrix multiply-accumulate operations
 * leveraging specialized GPU matrix cores on AMD's latest discrete GPUs. 'roc' being an AMD-specific
 * component belonging to the ROCm ecosystem, and WMMA stands for Wavefront Mixed precision Multiply Accumulate.
 *
 * rocWMMA leverages modern C++ techniques. It is templated for modularity and uses meta-programming paradigms to provide opportunities for customization
 * and compile-time inferences and optimizations. The API is seamless across supported CDNA and RDNA architectures. It is also portable with the Nvidia
 * nvcuda::wmma library, allowing those users to easily migrate to the AMD platform.
 *
 * The API is implemented as GPU device code which empowers users with direct use of GPU matrix cores, right from their kernel code.
 * Major benefits include kernel-level control which allows authoring flexibility and accessibility to compiler optimization passes in-situ
 * with other device code. Users can therefore decide when and where kernel run-time launches are required, which is not dictated by the API.
 *
 * rocWMMA's API facilitates the decomposition of matrix multiply-accumulate problems into discretized blocks (also known as fragments) and enables
 * parallelization of block-wise operations across multiple GPU wavefronts. The programmer's perspective is simplified to wavefront handling of fragments,
 * whereas individual threads are handled internally. This can allow for faster development times and a more seamless experience across multiple architectures.
 * API functions include data loading and storing, matrix multiply-accumulate and helper transforms that operate on data fragment abstractions. Moreover, data movement
 * between global and local memory can be done cooperatively amongst the wavefronts in a threadblock to enable data sharing and re-use. Matrix multiply-accumulate
 * functionality supports mixed precision inputs and outputs with native fixed-precision accumulation.
 *
 * Supporting code is required for GPU device management and kernel invocation. The kernel code samples and tests provided are built and launched via
 * the Heterogeneous-Compute Interface for Portability (HIP) ecosystem within ROCm.
 *
 * This library is an ongoing Work-In-Progress (WIP).
 *
 * For more documentation, please visit https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html.
 *
*/

namespace rocwmma
{
    //! @defgroup Rocwmma rocWMMA Public API
    //!
    //! @brief rocWMMA objects and API function definitions.
    //! @{

    //! @struct row_major
    //! @brief Meta-tag indicating 2D in-memory data layout as row major.
    struct row_major
    {
    };

    //! @struct col_major
    //! @brief Meta-tag indicating 2D in-memory data layout as column major.
    struct col_major
    {
    };

    //! @struct matrix_a
    //! @brief Meta-tag indicating data context is input Matrix A.
    struct matrix_a
    {
    };

    //! @struct matrix_b
    //! @brief Meta-tag indicating data context is input Matrix B.
    struct matrix_b
    {
    };

    //! @struct accumulator
    //! @brief Meta-tag indicating data context is Accumulator (also used as Matrix C / D).
    struct accumulator
    {
    };

    //! @struct layout_t
    //! @brief Runtime data layout tags
    //! @var mem_row_major
    //! @var mem_col_major
    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };

    //! @class fragment
    //! @brief rocWMMA fragment class. This is the primary object used in block-wise decomposition of the matrix multiply-accumulate (mma)
    //! problem space. In general, fragment data is associated with a matrix context (matrix_a, matrix_b or accumulator), a block size (BlockM/N/K),
    //! a datatype (e.g. single-precision float, etc.) and an in-memory 2D layout (e.g. row_major or col_major). These fragment properties are used
    //! to define how data is handled and stored locally, and to drive API implementations for loading / storing, mma and transforms. Fragment abstractions are
    //! designed to promote a simple wavefront programming model, which can accelerate development time. Internal thread-level details are handled by rocWMMA
    //! which frees the user to focus on wavefront block-wise decomposition. Written purely in device code, the programmer can use this object in their own
    //! device kernels.
    //!
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    //!
    //! @note Fragments are stored in packed registers, however vector elements have no guaranteed order or locality.
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT = void>
    class __align__(4) fragment
    {
    public:
        //! Input / output traits specific to AMDGCN architecture
        using IOTraits =
            typename IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::IOTraits;
        struct Traits
        {
        private:
            //! The packed type for element data
            using PackedElementT = typename PackTraits<DataT>::PackedT;

            //! The unpacked type for element data
            using UnpackedElementT = typename PackTraits<DataT>::UnpackedT;

        public:
            //! Unpacked data access view
            using AccessT = VecT<UnpackedElementT, IOTraits::UnpackedSize>;

            //! Packed data storage view
            using StorageT = VecT<PackedElementT, IOTraits::PackedSize>;

            constexpr static uint32_t Size = IOTraits::UnpackedSize;

            static_assert(IOTraits::PackedVRegCount >= 1,
                          "Fragments must occupy at least one packed register");
            static_assert(IOTraits::UnpackedSize % IOTraits::PackedSize == 0,
                          "Unable to pack fragment elements");
        };

        ROCWMMA_DEVICE           fragment() = default;
        ROCWMMA_DEVICE           fragment(const fragment& other);
        ROCWMMA_DEVICE fragment& operator=(const fragment& other);

        //! @param index Element index
        //! @returns Mutable unpacked element accessor at given index
        ROCWMMA_DEVICE inline DataT& operator[](uint32_t index);
        //! @param index Element index
        //! @returns Immutable unpacked element accessor at given index
        ROCWMMA_DEVICE inline DataT const& operator[](uint32_t index) const;
        //! @returns Mutable packed storage vector accessor
        ROCWMMA_DEVICE inline typename Traits::StorageT& operator*();
        //! @returns Immutable packed storage vector accessor
        ROCWMMA_DEVICE inline typename Traits::StorageT const& operator*() const;

        //! @returns The geometric height of fragment
        ROCWMMA_DEVICE constexpr static inline uint32_t height();
        //! @returns The geometric width of fragment
        ROCWMMA_DEVICE constexpr static inline uint32_t width();
        //! @returns The leading block dimension (non-K)
        ROCWMMA_DEVICE constexpr static inline uint32_t blockDim();
        //! @returns The k dimension
        ROCWMMA_DEVICE constexpr static inline uint32_t kDim();
        //! @returns The size of the unpacked elements vector
        ROCWMMA_DEVICE constexpr static inline uint32_t size();

        //! Internal data storage views. Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT             mStorage; // Packed
            typename Traits::AccessT              mAccess; // Unpacked
            typename Traits::AccessT::Native_vec_ x; // Nuanced access
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };

        // For compatibility
        constexpr static uint32_t num_elements = Traits::Size;
        using element_type                     = DataT;
    };

    //! Fills the entire fragment with the desired value.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param value Fill value of type DataT
    //! @tparam Matrix Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                      DataT                                                          value);

    //! Loads the entire fragment from the data pointer according to its matrix and data layout contexts. Data pointer may point to either local or global memory.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global or local memory
    //! @param ldm Leading dimension size
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT In-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                         const DataT*                                                   data,
                         uint32_t                                                       ldm);

    //! Loads the entire fragment from the data pointer according to its matrix layout and data layout contexts.
    //! Data pointer may point to either local or global memory. This overload provides a run-time ability to choose the data layout of the target fragment.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param layout Data layout
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                         const DataT*                                      data,
                                         uint32_t                                          ldm,
                                         layout_t                                          layout);

    //! Stores the entire fragment to the data pointer according to its matrix and data layouts. Data pointer may point to either local or global memory.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                               data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
                          uint32_t                                                             ldm);

    //! Stores the entire fragment to the data pointer according to its matrix layout. Data pointer may point to either local or global memory.
    //! This overload provides a run-time ability to choose the data layout of the target fragment.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param layout Data layout
    //! @tparam MatrixT Fragment context
    //! @tparam BlockM/N/K Block dimensions
    //! @tparam DataT Datatype
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                  data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                          uint32_t                                                ldm,
                          layout_t                                                layout);

    //! Performs the Multiply-Accumulate operation on the fragments A, B, C and D (D = A * B + C)
    //! @param d Accumulator output D
    //! @param a Input fragment A
    //! @param b Input fragment B
    //! @param c Input accumulator fragment C
    //! @tparam BlockM/N/K block dimensions
    //! @tparam InputT Datatype of input frags A and B
    //! @tparam ComputeT Datatype of accumulator fragment C / D
    //! @tparam LayoutA/B/C/D In-memory layout of frag as col_major or row_major
    //! @note Frag c = d is valid
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
                 fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutC> const& c);

    //! Synchronization point for all wavefronts in a workgroup. Guarantees pending reads / writes to LDS are flushed.
    ROCWMMA_DEVICE void synchronize_workgroup();

    /** @}*/
} // namespace rocwmma

#include "rocwmma_impl.hpp"

#endif // ROCWMMA_API_HPP
