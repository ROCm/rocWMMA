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

function(add_custom_test TEST_TARGET TEST_SOURCE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} wmma OpenMP::OpenMP_CXX gtest_main)
  target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_TESTS)
  gtest_discover_tests(${TEST_TARGET})
endfunction()

function(add_custom_test_with_rocblas TEST_TARGET TEST_SOURCE)
  add_custom_test(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} roc::rocblas)
  target_compile_definitions(${TEST_TARGET} PRIVATE WMMA_VALIDATE_WITH_ROCBLAS)
endfunction()

add_custom_test(LoadStoreMatrixSyncTest LoadStoreMatrixSyncTest.cpp)
add_custom_test(FillFragmentTest FillFragmentTest.cpp)
add_custom_test(MappingUtilTest MappingUtilTest.cpp)
add_custom_test(LayoutTest LayoutTest.cpp)

if(WMMA_VALIDATE_WITH_ROCBLAS)
  add_custom_test_with_rocblas(MmaSyncTest-rocBLAS MmaSyncTest.cpp)
else()
  add_custom_test(MmaSyncTest MmaSyncTest.cpp)
endif()

add_executable(MmaSyncTest-bench MmaSyncTest.cpp)
target_link_libraries(MmaSyncTest-bench wmma)
