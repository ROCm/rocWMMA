###############################################################################
 #
 # MIT License
 #
 # Copyright 2021-2022 Advanced Micro Devices, Inc.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

include( CMakeDependentOption )

set(ROCWMMA_TEST_UNIT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Unit common test sources
set(UnitCommonSources ${ROCWMMA_COMMON_TEST_SOURCES}
                      ${CMAKE_CURRENT_SOURCE_DIR}/unit_kernel_base.cpp)

set(ContaminationTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_a_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_a_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_a_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_a_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_a_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_acc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_acc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_acc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_acc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_acc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_b_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_b_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_b_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_b_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/load_contamination_b_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_a_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_a_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_a_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_a_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_a_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_acc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_acc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_acc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_acc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_acc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_b_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_b_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_b_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_b_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/contamination_test/store_contamination_b_256.cpp
                     )

set(LayoutTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/col_layout_16x16_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/col_layout_16x16_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/col_layout_32x32_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/col_layout_32x32_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/colnt_layout_16x16_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/colnt_layout_16x16_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/colnt_layout_32x32_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/colnt_layout_32x32_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/row_layout_16x16_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/row_layout_16x16_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/row_layout_32x32_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/row_layout_32x32_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/rownt_layout_16x16_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/rownt_layout_16x16_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/rownt_layout_32x32_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_test/rownt_layout_32x32_t.cpp
                       )

set(MapUtilTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_m_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_m_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_m_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_m_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_n_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_n_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_n_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_block_to_matrix_override_n_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_m_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_m_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_m_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_m_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_n_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_n_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_n_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_matrix_to_data_override_n_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_thread_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_thread_to_matrix_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_thread_to_matrix_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_thread_to_matrix_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_thread_to_matrix_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_wave_to_matrix_16x16_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_wave_to_matrix_16x16_t.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_wave_to_matrix_32x32_n.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_util_test/map_wave_to_matrix_32x32_t.cpp
                       )

set(LoadStoreMatrixSyncTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_a_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_a_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_a_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_a_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_a_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_acc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_acc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_acc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_acc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_acc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_b_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_b_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_b_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_b_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_sync_test/load_store_matrix_sync_b_256.cpp
                    )

set(LoadStoreMatrixCoopSyncTestSources ${UnitCommonSources}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_a_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_a_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_a_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_a_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_a_256.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_acc_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_acc_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_acc_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_acc_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_acc_256.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_b_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_b_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_b_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_b_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/load_store_matrix_coop_sync_test/load_store_matrix_coop_sync_b_256.cpp
                 )

set(FillFragmentTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_a_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_a_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_a_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_a_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_a_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_acc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_acc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_acc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_acc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_acc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_b_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_b_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_b_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_b_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/fill_fragment_test/fill_fragment_b_256.cpp
                       )

set(VectorIteratorTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/vector_iterator_test/vector_iterator.cpp
                       )

# Benchmark Unit
add_rocwmma_test(contamination_test ${ContaminationTestSources})
add_rocwmma_test(layout_test ${LayoutTestSources})
add_rocwmma_test(map_util_test ${MapUtilTestSources})
add_rocwmma_test(load_store_matrix_sync_test ${LoadStoreMatrixSyncTestSources})
add_rocwmma_test(load_store_matrix_coop_sync_test ${LoadStoreMatrixCoopSyncTestSources})
add_rocwmma_test(fill_fragment_test ${FillFragmentTestSources})
add_rocwmma_test(vector_iterator_test ${VectorIteratorTestSources})

