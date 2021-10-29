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

include(GoogleTest)

# Regular gtests that don't need special definitions for validation
function(add_wmma_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} wmma OpenMP::OpenMP_CXX gtest_main)
  target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemm)
  if(WMMA_BUILD_ASSEMBLY)
    foreach(file_name ${TEST_SOURCE})
      add_custom_command(TARGET ${TEST_TARGET}
                         POST_BUILD
                         COMMAND make ARGS ${file_name}.s
                         COMMAND ${CMAKE_COMMAND} -E copy
                           "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TEST_TARGET}.dir/${file_name}.s"
                           "${CMAKE_CURRENT_BINARY_DIR}/${file_name}.s"
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endfunction()

# Tests with additional validation using cpu implementation
function(add_wmma_validation_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})
  add_wmma_test(${TEST_TARGET} ${TEST_SOURCE})
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_TESTS)
endfunction()

# Tests with additional validation using rocblas implementation
function(add_wmma_validation_test_with_rocblas TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})
  add_wmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} roc::rocblas)
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_WITH_ROCBLAS)
endfunction()

# Unit tests
add_wmma_test(LoadStoreMatrixSyncTest LoadStoreMatrixSyncTest.cpp)
add_wmma_test(FillFragmentTest FillFragmentTest.cpp)
add_wmma_test(MappingUtilTest MappingUtilTest.cpp)
add_wmma_test(LayoutTest LayoutTest.cpp)
add_wmma_test(VectorIteratorTest VectorIteratorTest.cpp)
add_wmma_test(ContaminationTest ContaminationTest.cpp)

# GEMM Tests
set(GemmCommonSources HipDevice.cpp gemm/GemmKernelBase.cpp)

set(MmaSyncTestSources ${GemmCommonSources}
                       gemm/test/MmaSyncTest_16x16_NN.cpp
                       gemm/test/MmaSyncTest_16x16_NT.cpp
                       gemm/test/MmaSyncTest_16x16_TN.cpp
                       gemm/test/MmaSyncTest_16x16_TT.cpp
                       gemm/test/MmaSyncTest_32x32_NN.cpp
                       gemm/test/MmaSyncTest_32x32_NT.cpp
                       gemm/test/MmaSyncTest_32x32_TN.cpp
                       gemm/test/MmaSyncTest_32x32_TT.cpp)

set(MmaSyncMultiTestSources ${GemmCommonSources}
                       gemm/test/MmaSyncMultiTest_16x16_NN.cpp
                       gemm/test/MmaSyncMultiTest_16x16_NT.cpp
                       gemm/test/MmaSyncMultiTest_16x16_TN.cpp
                       gemm/test/MmaSyncMultiTest_16x16_TT.cpp
                       gemm/test/MmaSyncMultiTest_32x32_NN.cpp
                       gemm/test/MmaSyncMultiTest_32x32_NT.cpp
                       gemm/test/MmaSyncMultiTest_32x32_TN.cpp
                       gemm/test/MmaSyncMultiTest_32x32_TT.cpp)

set(MmaSyncMultiLdsTestSources ${GemmCommonSources}
                       gemm/test/MmaSyncMultiLdsTest_16x16_NN_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NN_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NN_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NN_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NT_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NT_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NT_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_NT_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TN_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TN_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TN_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TN_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TT_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TT_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TT_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_16x16_TT_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NN_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NN_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NN_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NN_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NT_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NT_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NT_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_NT_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TN_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TN_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TN_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TN_2x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TT_1x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TT_1x2.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TT_2x1.cpp
                       gemm/test/MmaSyncMultiLdsTest_32x32_TT_2x2.cpp)

set(MmaSyncLdsTestSources ${GemmCommonSources}
                       gemm/test/MmaSyncLdsTest_16x16_NN.cpp
                       gemm/test/MmaSyncLdsTest_16x16_NT.cpp
                       gemm/test/MmaSyncLdsTest_16x16_TN.cpp
                       gemm/test/MmaSyncLdsTest_16x16_TT.cpp
                       gemm/test/MmaSyncLdsTest_32x32_NN.cpp
                       gemm/test/MmaSyncLdsTest_32x32_NT.cpp
                       gemm/test/MmaSyncLdsTest_32x32_TN.cpp
                       gemm/test/MmaSyncLdsTest_32x32_TT.cpp)

set(MmaSyncCoopLdsTestSources ${GemmCommonSources}
                       gemm/test/MmaSyncCoopLdsTest_16x16_NN.cpp
                       gemm/test/MmaSyncCoopLdsTest_16x16_NT.cpp
                       gemm/test/MmaSyncCoopLdsTest_16x16_TN.cpp
                       gemm/test/MmaSyncCoopLdsTest_16x16_TT.cpp
                       gemm/test/MmaSyncCoopLdsTest_32x32_NN.cpp
                       gemm/test/MmaSyncCoopLdsTest_32x32_NT.cpp
                       gemm/test/MmaSyncCoopLdsTest_32x32_TN.cpp
                       gemm/test/MmaSyncCoopLdsTest_32x32_TT.cpp)

# Benchmark GEMM
add_wmma_test(MmaSyncTest-bench ${MmaSyncTestSources})
add_wmma_test(MmaSyncMultiTest-bench ${MmaSyncMultiTestSources})
add_wmma_test(MmaSyncMultiLdsTest-bench ${MmaSyncMultiLdsTestSources})
add_wmma_test(MmaSyncLdsTest-bench ${MmaSyncLdsTestSources})
add_wmma_test(MmaSyncCoopLdsTest-bench ${MmaSyncCoopLdsTestSources})

# Validation GEMM
if(WMMA_VALIDATE_WITH_ROCBLAS)
  add_wmma_validation_test_with_rocblas(MmaSyncTest-rocBLAS ${MmaSyncTestSources})
  add_wmma_validation_test_with_rocblas(MmaSyncMultiTest-rocBLAS ${MmaSyncMultiTestSources})
  add_wmma_validation_test_with_rocblas(MmaSyncMultiLdsTest-rocBLAS ${MmaSyncMultiLdsTestSources})
  add_wmma_validation_test_with_rocblas(MmaSyncLdsTest-rocBLAS ${MmaSyncLdsTestSources})
  add_wmma_validation_test_with_rocblas(MmaSyncCoopLdsTest-rocBLAS ${MmaSyncCoopLdsTestSources})
else()
  add_wmma_validation_test(MmaSyncTest-cpu ${MmaSyncTestSources})
  add_wmma_validation_test(MmaSyncMultiTest-cpu ${MmaSyncMultiTestSources})
  add_wmma_validation_test(MmaSyncLdsTest-cpu ${MmaSyncLdsTestSources})
  add_wmma_validation_test(MmaSyncCoopLdsTest-cpu ${MmaSyncCoopLdsTestSources})
endif()
