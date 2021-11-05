###############################################################################
 #
 # MIT License
 #
 # Copyright 2021 Advanced Micro Devices, Inc.
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

cmake_dependent_option( WMMA_VALIDATE_WITH_ROCBLAS "Use rocBLAS for validation" ON "WMMA_BUILD_VALIDATION_TESTS" OFF )
cmake_dependent_option( WMMA_BENCHMARK_WITH_ROCBLAS "Include rocBLAS benchmark performance comparisons" OFF "WMMA_BUILD_BENCHMARK_TESTS" OFF )

if(WMMA_VALIDATE_WITH_ROCBLAS OR WMMA_BENCHMARK_WITH_ROCBLAS)
  find_package( rocblas REQUIRED PATHS /opt/rocm /opt/rocm/rocblas $ENV{ROCBLAS_DIR} )
endif()

set(WMMA_TEST_GEMM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

### GEMM tests that have rocBLAS support
# Use rocBLAS for validation
function(add_gemm_validation_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_wmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${WMMA_TEST_GEMM_INCLUDE_DIR})

  # Link to rocBLAS
  if(WMMA_VALIDATE_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_WITH_ROCBLAS)
  endif()
endfunction()

# Include rocBLAS performance benchmark
function(add_gemm_benchmark_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_wmma_benchmark_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${WMMA_TEST_GEMM_INCLUDE_DIR})

  # Link to rocBLAS
  if(WMMA_BENCHMARK_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_BENCHMARK_WITH_ROCBLAS)
  endif()
endfunction()

# GEMM common test sources
set(GemmCommonSources ${WMMA_COMMON_TEST_SOURCES}
                       GemmKernelBase.cpp)

set(MmaSyncTestSources ${GemmCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_16x16_NN.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_16x16_NT.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_16x16_TN.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_16x16_TT.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_32x32_NN.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_32x32_NT.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_32x32_TN.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncTest_32x32_TT.cpp)

set(MmaSyncLdsTestSources ${GemmCommonSources}
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_16x16_NN.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_16x16_NT.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_16x16_TN.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_16x16_TT.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_32x32_NN.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_32x32_NT.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_32x32_TN.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncLdsTest_32x32_TT.cpp)

set(MmaSyncCoopLdsTestSources ${GemmCommonSources}
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_16x16_NN.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_16x16_NT.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_16x16_TN.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_16x16_TT.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_32x32_NN.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_32x32_NT.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_32x32_TN.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncCoopLdsTest_32x32_TT.cpp)

set(MmaSyncMultiTestSources ${GemmCommonSources}
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_16x16_NN.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_16x16_NT.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_16x16_TN.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_16x16_TT.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_32x32_NN.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_32x32_NT.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_32x32_TN.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiTest_32x32_TT.cpp)

set(MmaSyncMultiLdsTestSources ${GemmCommonSources}
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NN_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NT_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TN_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TT_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NN_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NT_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TN_2x2.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TT_2x2.cpp)

if(WMMA_BUILD_EXTENDED_TESTS)
  set(MmaSyncMultiLdsTestSources ${MmaSyncMultiLdsTestSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NN_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NN_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NN_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NT_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NT_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_NT_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TN_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TN_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TN_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TT_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TT_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_16x16_TT_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NN_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NN_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NN_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NT_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NT_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_NT_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TN_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TN_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TN_2x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TT_1x1.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TT_1x2.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test/MmaSyncMultiLdsTest_32x32_TT_2x1.cpp)
endif()

### GEMM tests are built as either benchmark or validation

# Benchmark GEMM tests
if(WMMA_BUILD_BENCHMARK_TESTS)
  add_gemm_benchmark_test(MmaSyncTest-bench ${MmaSyncTestSources})
  add_gemm_benchmark_test(MmaSyncMultiTest-bench ${MmaSyncMultiTestSources})
  add_gemm_benchmark_test(MmaSyncMultiLdsTest-bench ${MmaSyncMultiLdsTestSources})
  add_gemm_benchmark_test(MmaSyncLdsTest-bench ${MmaSyncLdsTestSources})
  add_gemm_benchmark_test(MmaSyncCoopLdsTest-bench ${MmaSyncCoopLdsTestSources})
endif()

# Validation GEMM tests
if(WMMA_BUILD_VALIDATION_TESTS)
  add_gemm_validation_test(MmaSyncTest-validate ${MmaSyncTestSources})
  add_gemm_validation_test(MmaSyncMultiTest-validate ${MmaSyncMultiTestSources})
  add_gemm_validation_test(MmaSyncMultiLdsTest-validate ${MmaSyncMultiLdsTestSources})
  add_gemm_validation_test(MmaSyncLdsTest-validate ${MmaSyncLdsTestSources})
  add_gemm_validation_test(MmaSyncCoopLdsTest-validate ${MmaSyncCoopLdsTestSources})
endif()
