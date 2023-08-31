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
#ifndef ROCWMMA_COOP_API_HPP
#define ROCWMMA_COOP_API_HPP

#include "rocwmma.hpp"

/**
 * ROCWMMACoop complements the ROCWMMA API with support for cooperative operations.
 * Operations may be split into smaller unit work items that are assigned to waves
 * in a collaborative pool using a round-robin fashion.
 *
 * From each wave's perspective, their data responsibility determines the cohesiveness
 * of the data in wave's fragment. Cooperative operations are designed to be used
 * in conjunction with each other to cooperatively move data from a single full fragment,
 * as is common to do in GEMM algorithms. This API facilitates this functionality.
 *
 * \n
 * **load_matrix_coop_sync / store_matrix_coop_sync**
 *
 * Loads data from memory according to Matrix Layout.
 * Matrix A layout loads / stores matrix columns in the K direction
 * (Matrix A = M x K, fragA = BlockM x BlockK)
 * Matrix B layout loads / stores matrix rows in the K direction
 * (Matrix B = K x N, fragB = BlockK x BlockN)
 * Matrix C layout loads / stores matrix rows in vector width of 4
 * (Matrix C = M x N, fragAcc = BlockM x BlockN)
 *
 * Fragments are stored in packed registers in optimal load / store patterns.
 * In-register elements have no guaranteed order, which have been optimized for loading / storing efficiency.
 */

namespace rocwmma
{

    //! Cooperative Load Matrix - Loads the entire fragment with data from memory address cooperatively across waves.
    //! Each cooperative wave is responsible in loading a portion of the final fragment.
    //! Note that the full fragment data is not cohesive for individual waves as they only load a piece of the data.
    //! This function may be paired with store_matrix_sync to move a single fragment collaboratively between memory locations.
    //!
    //! The full load is split into work items (splitCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until
    //! there are no more work items and the operation is completed.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param waveIndex Index assignment of current wave in collaboration
      \param waveCount Number of waves assigned for collaboration
      \param splitCount Number of work items to split the operation
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
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount,
                              uint32_t splitCount);

    //! Cooperative Load Matrix - Loads the entire fragment with data from memory address cooperatively across waves.
    //! Each cooperative wave is responsible in loading a portion of the final fragment.
    //! Note that the full fragment data is not cohesive for individual waves as they only load a piece of the data.
    //! This function may be paired with store_matrix_sync to move a single fragment collaboratively
    //! between memory locations.
    //!
    //! The full load is split into work items (default = waveCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param waveIndex Index assignment of current wave in collaboration
      \param waveCount Number of waves assigned for collaboration
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
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount);

    //! Cooperative Load Matrix - Loads the entire fragment with data from memory address cooperatively across waves.
    //! Each cooperative wave is responsible in loading a portion of the final fragment.
    //! Note that the full fragment data is not cohesive for individual waves as they only load a piece of the data.
    //! This function may be paired with store_matrix_sync to move a single fragment collaboratively
    //!between memory locations.
    //!
    //! The full load is split into work items.
    //! This overload is conducive to GEMM workload where
    //! matrix_a fragments collaboratively load common data with other waves in the same row. Likewise,
    //! matrix_b fragments collaboratively load common data with other waves in the same column.
    //! Workload is split evenly across all waves in collaborative dimension.
    //! Split count = wave count = workgroupDim<1> (matrix_a) | workgroupDim<0> (matrix_b)
    //! Wave index = waveCoord<1> (matrix_a) | waveCoord<0> (matrix_b)
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
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag,
                              const DataT*                                                  data,
                              uint32_t                                                      ldm);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! Note that the full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! The full store is split into work items (splitCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param waveIndex Index assignment of current wave in collaboration
      \param waveCount Number of waves assigned for collaboration
      \param splitCount Number of work items to split the operation
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
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm,
        uint32_t                                                            waveIndex,
        uint32_t                                                            waveCount,
        uint32_t                                                            splitCount);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment. Note that the full
    //! fragment data is not required to be cohesive for individual waves as they only store a piece of the data.
    //! This function may be paired with load_matrix_sync to move a single fragment collaboratively between memory locations.
    //!
    //! The full store is split into work items (default = waveCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    /*!
      \param frag Fragment of type MatrixT with its associated block sizes, data type and layout
      \param data Data pointer to global/local memory
      \param ldm Leading dimension size
      \param waveIndex Index assignment of current wave in collaboration
      \param waveCount Number of waves assigned for collaboration
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
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm,
        uint32_t                                                            waveIndex,
        uint32_t                                                            waveCount);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! Note that the full fragment data is not required to be cohesive for individual waves as they only store a piece of the data.
    //! This function may be paired with load_matrix_sync to move a single fragment collaboratively between memory locations.
    //!
    //! The full store is split into work items. This overload is conducive to GEMM workload where
    //! matrix_a fragments collaboratively use common data with other waves in the same row. Likewise,
    //! matrix_b fragments collaboratively use common data with other waves in the same column.
    //! Workload is split evenly across all waves in collaborative dimension.
    //!
    //! Split count = wave count = workgroupDim<1> (matrix_a) | workgroupDim<0> (matrix_b)
    //! Wave index = waveCoord<1> (matrix_a) | waveCoord<0> (matrix_b)
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
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                              data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag,
        uint32_t                                                            ldm);

} // namespace rocwmma

#include "rocwmma_coop_impl.hpp"

#endif // ROCWMMA_COOP_API_HPP
