# #############################################################################
# Copyright 2021 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

include(GoogleTest)

# Regular gtests that don't need special definitions for validation
function(add_wmma_test TEST_TARGET TEST_SOURCE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} wmma OpenMP::OpenMP_CXX gtest_main)
  target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  gtest_discover_tests(${TEST_TARGET})
  if(WMMA_BUILD_ASSEMBLY)
    add_custom_command(TARGET ${TEST_TARGET}
                       POST_BUILD
                       COMMAND make ARGS ${TEST_SOURCE}.s
                       COMMAND ${CMAKE_COMMAND} -E copy
                           "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TEST_TARGET}.dir/${TEST_SOURCE}.s"
                           "${CMAKE_CURRENT_BINARY_DIR}/${TEST_TARGET}.s"
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction()

# Tests with additional validation using cpu implementation
function(add_wmma_validation_test TEST_TARGET TEST_SOURCE)
  add_wmma_test(${TEST_TARGET} ${TEST_SOURCE})
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_TESTS)
endfunction()

# Tests with additional validation using rocblas implementation
function(add_wmma_validation_test_with_rocblas TEST_TARGET TEST_SOURCE)
  add_wmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} roc::rocblas)
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_WITH_ROCBLAS)
endfunction()

add_wmma_test(LoadStoreMatrixSyncTest LoadStoreMatrixSyncTest.cpp)
add_wmma_test(FillFragmentTest FillFragmentTest.cpp)
add_wmma_test(MappingUtilTest MappingUtilTest.cpp)
add_wmma_test(LayoutTest LayoutTest.cpp)
add_wmma_test(VectorIteratorTest VectorIteratorTest.cpp)
add_wmma_test(ContaminationTest ContaminationTest.cpp)

add_wmma_test(MmaSyncTest-bench MmaSyncTest.cpp)
add_wmma_test(MmaSyncTestLds-bench MmaSyncTestLds.cpp)
add_wmma_test(MmaSyncTestCoopLds-bench MmaSyncTestCoopLds.cpp)

if(WMMA_VALIDATE_WITH_ROCBLAS)
  add_wmma_validation_test_with_rocblas(MmaSyncTest-rocBLAS MmaSyncTest.cpp)
  add_wmma_validation_test_with_rocblas(MmaSyncTestLds-rocBLAS MmaSyncTestLds.cpp)
  add_wmma_validation_test_with_rocblas(MmaSyncTestCoopLds-rocBLAS MmaSyncTestCoopLds.cpp)
else()
  add_wmma_validation_test(MmaSyncTest-cpu MmaSyncTest.cpp)
  add_wmma_validation_test(MmaSyncTestLds-cpu MmaSyncTestLds.cpp)
  add_wmma_validation_test(MmaSyncTestCoopLds-cpu MmaSyncTestCoopLds.cpp)
endif()
