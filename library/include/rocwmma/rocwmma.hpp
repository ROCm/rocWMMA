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
#ifndef ROCWMMA_HPP
#define ROCWMMA_HPP

#include <type_traits>

#include "internal/io_config.hpp"
#include "internal/io_traits.hpp"
#include "internal/types.hpp"

/**
 * \mainpage
 *
 * ROCWMMA is a C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications
 * leveraging AMD's GPU hardware matrix cores through HIP.
 * Specifically, the library enhances the portability of CUDA WMMA code to
 * AMD's heterogeneous platform and provides an interface to use underlying
 * hardware matrix multiplication (MFMA) units.
 * The ROCWMMA API exposes memory and MMA (Matrix Multiply Accumulate) functions
 * that operate on blocks, or 'fragments' of data appropriately sized for
 * warp (thread block) execution.
 * ROCWMMA code is templated for componentization and for providing ability to
 * make compile-time optimizations based on available meta-data.
 * This library is an ongoing Work-In-Progress (WIP).
 *
 *
 * **Supported Datatypes**
 *  - Native Data Types
 *      - float = f32
 *      - double = f64
 *      - _Float16 = f16
 *      - int8
 *      - uint8
 *      - int16
 *      - int32
 *      - uint32
 *
 *
 *  - Non-Native Data Types
 *      - h16 = __half
 *      - bf16 = bfloat16
 *
 * **Supported Thread Block Sizes**
 *
 * TBlockX  | TBlockY   |
 * :-------:|:---------:|
 * 64       |   1       |
 * 64       |   2       |
 * 64       |   4       |
 * 64       |   8       |
 * 128      |   1       |
 * 128      |   2       |
 * 256      |   1       |
 *
 * @note TBlockX must be a multiple of 64
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
 *
 *
 * **Data Types <Ti / To / Tc> = <InputType / OutputType / ComputeType >**
 * \n
 * **MFMA Block Size = <BlockM, BlockN, BlockK>**
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
 *
 * **LDS Support/Requirements(Only applicable to ROCWMMA v0.3 and below)**
 *
 * Required LDS space is calculated by
 *  - LDSBytes = max(BlockM * blockDim.y, BlockN * blockDim.x / 64) * BlockK * sizeof(InputT)
 *
 *
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
 * Matrix C layout loads / stores matrix rows in vector width of 4
 * (Matrix C = M x N, fragAcc = BlockM x BlockN)
 *
 * @note Fragments are stored in packed registers, however elements have no guaranteed order.
 *
 * \n
 * **mma_sync**
 *
 * MFMA accumulation is performed with fragment data. Fragment A cols are multiplied
 * with Fragment B rows and added to the accumulator fragment.
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
 * \ingroup rocwmma
 * \defgroup ROCWMMA APIs
 *
 * @brief ROCWMMA Fragment and its API function definitions.
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
            using PackedElementT   = typename detail::PackTraits<DataT>::PackedT;
            using UnpackedElementT = typename detail::PackTraits<DataT>::UnpackedT;
            using IOTraits =
                typename io_config<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>::IOTraits;

        public:
            using AccessT  = VecT<UnpackedElementT, IOTraits::UnpackedSize>;
            using StorageT = VecT<PackedElementT, IOTraits::PackedSize>;

            static_assert(IOTraits::PackedVRegCount >= 1,
                          "Fragments must occupy at least one packed register");
            static_assert(IOTraits::UnpackedSize % IOTraits::PackedSize == 0,
                          "Unable to pack fragment elements");
        };

        using IOConfig = io_config<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>;

        __device__           fragment() = default;
        __device__           fragment(const fragment& other);
        __device__ fragment& operator=(const fragment& other);

        // Accessors
        __device__ inline DataT&                           operator[](uint32_t index);
        __device__ inline DataT const&                     operator[](uint32_t index) const;
        __device__ inline typename Traits::StorageT&       operator*();
        __device__ inline typename Traits::StorageT const& operator*() const;

        // Traits
        __device__ constexpr static inline uint32_t blockDim();
        __device__ constexpr static inline uint32_t kDim();
        __device__ constexpr static inline uint32_t size();

        // Compatibility with nvcuda::wmma
        union
        {
            typename Traits::StorageT mStorage; // Packed
            typename Traits::AccessT  mAccess; // Unpacked
            typename Traits::AccessT  x; // Nuanced access
            static_assert(sizeof(typename Traits::AccessT) == sizeof(typename Traits::StorageT),
                          "Storage type and access type should be views into the same raw data");
        };
        constexpr static uint32_t num_elements = Traits::AccessT::Size;
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
    __device__ void
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
    __device__ void
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
    __device__ void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
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
    __device__ void
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
    __device__ void store_matrix_sync(DataT*                                                  data,
                                      fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                                      uint32_t                                                ldm,
                                      layout_t layout);

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
    __device__ void
        mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutD>&       d,
                 fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA> const&      a,
                 fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB> const&      b,
                 fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutC> const& c);

    //! Synchronization point for all wavefronts in a workgroup.
    __device__ void synchronize_workgroup();

} // namespace rocwmma

#include "rocwmma_impl.hpp"

#endif // ROCWMMA_HPP
