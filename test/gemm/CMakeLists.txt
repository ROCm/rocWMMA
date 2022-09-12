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

cmake_dependent_option( ROCWMMA_VALIDATE_WITH_ROCBLAS "Use rocBLAS for validation" OFF "ROCWMMA_BUILD_VALIDATION_TESTS" OFF )
cmake_dependent_option( ROCWMMA_BENCHMARK_WITH_ROCBLAS "Include rocBLAS benchmark performance comparisons" OFF "ROCWMMA_BUILD_BENCHMARK_TESTS" OFF )

if(ROCWMMA_VALIDATE_WITH_ROCBLAS OR ROCWMMA_BENCHMARK_WITH_ROCBLAS)
  find_package( rocblas REQUIRED PATHS /opt/rocm /opt/rocm/rocblas $ENV{ROCBLAS_DIR} )
  rocm_package_add_dependencies("rocblas >= 2.32.0" COMPONENT tests)
endif()

set(ROCWMMA_TEST_GEMM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

### GEMM tests that have rocBLAS support
# Use rocBLAS for validation
function(add_gemm_validation_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_rocwmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${ROCWMMA_TEST_GEMM_INCLUDE_DIR})

  # Link to rocBLAS
  if(ROCWMMA_VALIDATE_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCWMMA_VALIDATE_WITH_ROCBLAS)
  endif()
endfunction()

# Include rocBLAS performance benchmark
function(add_gemm_benchmark_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_rocwmma_benchmark_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${ROCWMMA_TEST_GEMM_INCLUDE_DIR})

  # Link to rocBLAS
  if(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCWMMA_BENCHMARK_WITH_ROCBLAS)
  endif()
endfunction()

# GEMM common test sources
set(GemmCommonSources ${ROCWMMA_COMMON_TEST_SOURCES}
                      ${CMAKE_CURRENT_SOURCE_DIR}/gemm_kernel_base.cpp)

set(BarrierTestSources ${GemmCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/barrier_test.cpp)

set(MmaSyncTestSources ${GemmCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_16x16_nn.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_16x16_nt.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_16x16_tn.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_16x16_tt.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_32x32_nn.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_32x32_nt.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_32x32_tn.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_test_32x32_tt.cpp
                       )

set(MmaSyncMultiTestSources ${GemmCommonSources}
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nn_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nt_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tn_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tt_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nn_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nt_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tn_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tt_1x1.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nn_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nt_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tn_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tt_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nn_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nt_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tn_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tt_2x2.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nn_4x4.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nt_4x4.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tn_4x4.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tt_4x4.cpp
			                      )

if(ROCWMMA_BUILD_EXTENDED_TESTS)
  set(MmaSyncMultiTestSources ${MmaSyncMultiTestSources}
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nn_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nt_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tn_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tt_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nn_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_nt_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tn_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_16x16_tt_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nn_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nt_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tn_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tt_1x2.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nn_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_nt_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tn_2x1.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_test_32x32_tt_2x1.cpp

                              )
endif()

set(MmaSyncMultiLdsTestSources ${GemmCommonSources}
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nn_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nt_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tn_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tt_1x1.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nn_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nt_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tn_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tt_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_4x4.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_4x4.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_4x4.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_4x4.cpp
                               )

if(ROCWMMA_BUILD_EXTENDED_TESTS)
  set(MmaSyncMultiLdsTestSources ${MmaSyncMultiLdsTestSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nn_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nn_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nt_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nt_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tn_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tn_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tt_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tt_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nn_8x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_nt_8x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tn_8x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_16x16_tt_8x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nn_4x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_nt_4x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tn_4x4.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_test_32x32_tt_4x4.cpp
                                 )
endif()

set(MmaSyncCoopWgTestSources ${GemmCommonSources}
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nn_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nt_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tn_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tt_4x4.cpp
			                      )

if(ROCWMMA_BUILD_EXTENDED_TESTS)
  set(MmaSyncCoopWgTestSources ${MmaSyncCoopWgTestSources}
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nn_8x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_nt_8x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tn_8x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_16x16_tt_8x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nn_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_nt_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tn_4x4.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_test_32x32_tt_4x4.cpp
                                )
endif()

set(MmaSyncAdHocTestSources ${GemmCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_ad_hoc.cpp)

set(MmaSyncMultiAdHocTestSources ${GemmCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_ad_hoc.cpp)

set(MmaSyncMultiLdsAdHocTestSources ${GemmCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_multi_lds_ad_hoc.cpp)

set(MmaSyncCoopWgAdHocTestSources ${GemmCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mma_sync_coop_wg_ad_hoc.cpp)

### GEMM tests are built as either benchmark or validation

# Benchmark GEMM tests
if(ROCWMMA_BUILD_BENCHMARK_TESTS)
  add_gemm_benchmark_test(barrier_test-bench ${BarrierTestSources})
  add_gemm_benchmark_test(mma_sync_test-bench ${MmaSyncTestSources})
  add_gemm_benchmark_test(mma_sync_multi_test-bench ${MmaSyncMultiTestSources})
  add_gemm_benchmark_test(mma_sync_multi_lds_test-bench ${MmaSyncMultiLdsTestSources})
  add_gemm_benchmark_test(mma_sync_coop_wg_test-bench ${MmaSyncCoopWgTestSources})
  add_gemm_benchmark_test(mma_sync_ad_hoc_test-bench ${MmaSyncAdHocTestSources})
  add_gemm_benchmark_test(mma_sync_multi_ad_hoc_test-bench ${MmaSyncMultiAdHocTestSources})
  add_gemm_benchmark_test(mma_sync_multi_lds_ad_hoc_test-bench ${MmaSyncMultiLdsAdHocTestSources})
  add_gemm_benchmark_test(mma_sync_coop_wg_ad_hoc_test-bench ${MmaSyncCoopWgAdHocTestSources})
endif()

# Validation GEMM tests
if(ROCWMMA_BUILD_VALIDATION_TESTS)
  add_gemm_validation_test(barrier_test-validate ${BarrierTestSources})
  add_gemm_validation_test(mma_sync_test-validate ${MmaSyncTestSources})
  add_gemm_validation_test(mma_sync_multi_test-validate ${MmaSyncMultiTestSources})
  add_gemm_validation_test(mma_sync_multi_lds_test-validate ${MmaSyncMultiLdsTestSources})
  add_gemm_validation_test(mma_sync_coop_wg_test-validate ${MmaSyncCoopWgTestSources})
  add_gemm_validation_test(mma_sync_ad_hoc_test-validate ${MmaSyncAdHocTestSources})
  add_gemm_validation_test(mma_sync_multi_ad_hoc_test-validate ${MmaSyncMultiAdHocTestSources})
  add_gemm_validation_test(mma_sync_multi_lds_ad_hoc_test-validate ${MmaSyncMultiLdsAdHocTestSources})
  add_gemm_validation_test(mma_sync_coop_wg_ad_hoc_test-validate ${MmaSyncCoopWgAdHocTestSources})
endif()
