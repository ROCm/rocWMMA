/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#include "internal/io_config.hpp"
#include "internal/io_traits.hpp"
#include "internal/pack_util.hpp"
#include "internal/types.hpp"

/**
 * \mainpage
 *
 * ROCWMMA is a C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications
 * leveraging AMD's GPU hardware through HIP.
 * Specifically, the library enhances the portability of CUDA WMMA code to
 * AMD's heterogeneous platform and provides an interface to use underlying
 * hardware matrix multiplication.
 * The ROCWMMA API exposes memory and MMA (Matrix Multiply Accumulate) functions
 * that operate on blocks, or 'fragments' of data appropriately sized for
 * warp (thread block) execution.
 * ROCWMMA code is templated for componentization and for providing ability to
 * make compile-time optimizations based on available meta-data.
 * This library is an ongoing Work-In-Progress (WIP).
 *
 * **Supported Hardware**
 * - CDNA architecture: gfx908, gfx90a, gfx940, gfx941, gfx942 (gfx9)
 * - RDNA3 architecture: gfx1100, gfx1101, gfx1102 (gfx11)
 *
 * **Supported Wave Sizes**
 * - Wave 32 (gfx11 only)
 * - Wave 64 (gfx9 only)
 *
 * **Supported Datatypes (gfx9)**
 *  - Native Data Types
 *      - float = f32
 *      - double = f64 (*only on gfx90a, gfx940, gfx941 & gfx942)
 *      - _Float16 = f16
 *      - int8
 *
 *  - Non-Native Data Types
 *      - h16 = __half
 *      - bf16 = bfloat16
 *
 * **Supported Datatypes (gfx11)**
 *  - Native Data Types
 *      - _Float16 = f16
 *      - int8
 *
 *  - Non-Native Data Types
 *      - h16 = __half
 *      - bf16 = bfloat16
 *
 * **Supported Thread Block Sizes**
 * Total wave count of 4
 * TBlockX    | TBlockY   |
 * :---------:|:---------:|
 * WaveSize   |   1       |
 * WaveSize   |   2       |
 * WaveSize   |   4       |
 * WaveSize*2 |   1       |
 * WaveSize*2 |   2       |
 * WaveSize*4 |   1       |
 *
 * @note TBlockX must be a multiple of WaveSize
 *
 *
 * **Supported Matrix Layouts**
 *
 * Matrix Layout(N = col major, T = row major)
 *
 * LayoutA  |   LayoutB |   LayoutC |   LayoutD  |
 * :-------:|:---------:|:---------:|:----------:|
 *     N    |      N    |      N    |     N      |
 *     N    |      T    |      N    |     N      |
 *     T    |      N    |      N    |     N      |
 *     T    |      T    |      N    |     N      |
 *     N    |      N    |      T    |     T      |
 *     N    |      T    |      T    |     T      |
 *     T    |      N    |      T    |     T      |
 *     T    |      T    |      T    |     T      |
 *
 * **Data Types <Ti / To / Tc> = <InputType / OutputType / ComputeType >**
 * \n
 * **MMA Block Size = <BlockM, BlockN, BlockK>**
 * @note gfx11 only supports BlockM/N = 16
 * \n
 * Ti / To / Tc         |   BlockM    |   BlockN    |   BlockK
 * :-------------------:|:-----------:|:-----------:|:-----------:|
 * i8/i32/i32           |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * i8/i8/i32            |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f32/f32          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f16/f32          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * f16/f16/f16          |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/f32/f32       |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/__half/f32    |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * __half/__half/__half |    16       |      16     | Min:16,pow2 |
 * ^                    |    32       |      32     | Min:8, pow2 |
 * bf16/f32/f32         |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * bf16/bf16/f32        |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * bf16/bf16/bf16       |    16       |      16     | Min:8, pow2 |
 * ^                    |    32       |      32     | Min:4, pow2 |
 * f32/f32/f32          |    16       |      16     | Min:4, pow2 |
 * ^                    |    32       |      32     | Min:2, pow2 |
 * f64/f64/f64          |    16       |      16     | Min:4, pow2 |
 *
 *
 * \n
 * \n
 * **Fragment:**
 *
 * **fill_fragment**
 *
 * Broadcast a desired value to all elements in the fragment.
 *
 * \n
 * **load_matrix_sync / store_matrix_sync**
 *
 * Loads data from memory according to Matrix Layout.
 * Matrix A layout loads / stores matrix columns in the K direction
 * (Matrix A = M x K, fragA = BlockM x BlockK)
 * Matrix B layout loads / stores matrix rows in the K direction
 * (Matrix B = K x N, fragB = BlockK x BlockN)
 * Matrix C layout loads / stores matrix rows in the M direction
 * (Matrix C = M x N, fragAcc = BlockM x BlockN)
 *
 * @note Fragments are stored in packed registers, however elements have no guaranteed order.
 *
 * \n
 * **mma_sync**
 *
 * MMA is performed with fragment data. The outer product of Fragment A cols
 * with Fragment B rows are added back into the accumulator fragment.
 *
 * **synchronize_workgroup**
 * Synchronization point for all wavefronts in a workgroup.
 */

namespace rocwmma
{
    // Configuration profile used in rocwmma calls
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    using io_config = rocwmma::IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;

    /**
 * \defgroup Rocwmma ROCWMMA Public API
 *
 * @brief ROCWMMA Fragment and its API function definitions.
 * @{
 */

    /*! \class fragment
 *  \brief Definition of MFMA Fragment
 *
 * @tparam MatrixT - fragment context
 * @tparam BlockM/N/K - block dimensions
 * @tparam DataT - data type
 * @tparam DataLayout - in-memory layout as col_major or row_major
 *
 * PackedT - The type of the vector register holding packed element data
 * UnpackedT - The type of the vector register holding unpacked element data
 * IOTraits - Input/output traits specific to AMDGCN architecture
 * AccessT - Unpacked data storage
 * StorageT = Packed data storage required for MFMA
 *
 * @note Fragments are stored in packed registers, however elements have no guaranteed order.
 */
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout = void>
    class __align__(4) fragment
    {
    public:
        struct Traits
        {
        private:
            using PackedElementT   = typename PackTraits<DataT>::PackedT;
            using UnpackedElementT = typename PackTraits<DataT>::UnpackedT;
            using IOTraits =
                typename io_config<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>::IOTraits;

        public:
            using AccessT  = VecT<UnpackedElementT, IOTraits::UnpackedSize>;
            using StorageT = VecT<PackedElementT, IOTraits::PackedSize>;

            constexpr static uint32_t Size = IOTraits::UnpackedSize;

            static_assert(IOTraits::PackedVRegCount >= 1,
                          "Fragments must occupy at least one packed register");
            static_assert(IOTraits::UnpackedSize % IOTraits::PackedSize == 0,
                          "Unable to pack fragment elements");
        };

        using IOConfig = io_config<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;

        ROCWMMA_DEVICE           fragment() = default;
        ROCWMMA_DEVICE           fragment(const fragment& other);
        ROCWMMA_DEVICE fragment& operator=(const fragment& other);

        // Accessors
        ROCWMMA_DEVICE inline DataT&                           operator[](uint32_t index);
        ROCWMMA_DEVICE inline DataT const&                     operator[](uint32_t index) const;
        ROCWMMA_DEVICE inline typename Traits::StorageT&       operator*();
        ROCWMMA_DEVICE inline typename Traits::StorageT const& operator*() const;

        // Traits
        ROCWMMA_DEVICE constexpr static inline uint32_t blockDim();
        ROCWMMA_DEVICE constexpr static inline uint32_t kDim();
        ROCWMMA_DEVICE constexpr static inline uint32_t size();

        // Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT             mStorage; // Packed
            typename Traits::AccessT              mAccess; // Unpacked
            typename Traits::AccessT::Native_vec_ x; // Nuanced access
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };
        constexpr static uint32_t num_elements = Traits::Size;
        using element_type                     = DataT;
    };

    //! Fills the entire fragment with the desired value.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param value Value of type DataT.
      \tparam Matrix fragment context
      \tparam BlockM/N/K block dimensions
      \tparam DataT data type
      \tparam DataLayout in-memory layout as col_major or row_major
    */
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                      DataT                                                         value);

    //! Loads the entire fragment from the data pointer according to its matrix and data layouts. Data pointer may point to either local or global memory.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \tparam MatrixT fragment context
      \tparam BlockM/N/K block dimensions
      \tparam DataT data type
      \tparam DataLayout in-memory layout as col_major or row_major
    */
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                         const DataT*                                                  data,
                         uint32_t                                                      ldm);

    //! Loads the entire fragment from the data pointer according to its matrix layout.Data pointer may point to either local or global memory. This overload provides a run-time ability to choose the data layout of the target fragment.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param layout Matrix layout
      \tparam MatrixT fragment context
      \tparam BlockM/N/K block dimensions
      \tparam DataT data type
      \tparam DataLayout in-memory layout as col_major or row_major
    */
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                         const DataT*                                      data,
                                         uint32_t                                          ldm,
                                         layout_t                                          layout);

    //! Stores the entire fragment to the data pointer according to its matrix and data layouts. Data pointer may point to either local or global memory.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \tparam MatrixT fragment context
      \tparam BlockM/N/K block dimensions
      \tparam DataT data type
      \tparam DataLayout in-memory layout as col_major or row_major
    */
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayout>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                              data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
                          uint32_t                                                            ldm);

    //!  Stores the entire fragment to the data pointer according to its matrix layout. Data pointer may point to either local or global memory. This overload provides a run-time ability to choose the data layout of the target fragment.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param layout Data layout
      \tparam MatrixT fragment context
      \tparam BlockM/N/K block dimensions
      \tparam DataT data type
      \tparam DataLayout in-memory layout as col_major or row_major
    */
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                  data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                          uint32_t                                                ldm,
                          layout_t                                                layout);

    //! Performs the Multiply-Accumulate operation on the fragments A, B, C and D(D = A * B + C)
    /*!
      \param d Accumulator output D
      \param a Input fragment A
      \param b Input fragment B
      \param c Input accumulator fragment C
      \tparam BlockM/N/K block dimensions
      \tparam InputT data type of input frags A and B
      \tparam ComputeT data type of accumulator fragment C / D
      \tparam LayoutA in-memory layout of frag A as col_major or row_major
      \tparam LayoutB in-memory layout of frag B as col_major or row_major
      \note Frag c = d is valid
    */
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

    //! Synchronization point for all wavefronts in a workgroup.
    ROCWMMA_DEVICE void synchronize_workgroup();

    /** @}*/
} // namespace rocwmma

#include "rocwmma_impl.hpp"

#endif // ROCWMMA_API_HPP
