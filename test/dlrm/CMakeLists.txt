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

 set(ROCWMMA_TEST_DLRM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

 function(add_dlrm_validation_test TEST_TARGET TEST_SOURCE)
   list(APPEND TEST_SOURCE ${ARGN})

   # Create target
   add_rocwmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})

   # Add dlrm include directory
   target_include_directories(${TEST_TARGET} PRIVATE ${WMMA_TEST_DLRM_INCLUDE_DIR})
 endfunction()

 # Include rocBLAS performance benchmark
 function(add_dlrm_benchmark_test TEST_TARGET TEST_SOURCE)
   list(APPEND TEST_SOURCE ${ARGN})

   # Create target
   add_rocwmma_benchmark_test(${TEST_TARGET} ${TEST_SOURCE})

   # Add dlrm include directory
   target_include_directories(${TEST_TARGET} PRIVATE ${WMMA_TEST_DLRM_INCLUDE_DIR})
 endfunction()

 set(DlrmCommonSources ${WMMA_COMMON_TEST_SOURCES}
                       dlrm_kernel_base.cpp)

 set(DlrmDotTestSources ${DlrmCommonSources}
                        ${CMAKE_CURRENT_SOURCE_DIR}/test/dlrm_dot_test.cpp)

  set(DlrmDotLdsTestSources ${DlrmCommonSources}
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/DlrmDotLdsTest.cpp)

 # Benchmark DLRM tests
 if (WMMA_BUILD_BENCHMARK_TESTS)
     add_dlrm_benchmark_test(dlrm_dot_test-bench ${DlrmDotTestSources})
     add_dlrm_benchmark_test(dlrm_dot_lds_test-bench ${DlrmDotLdsTestSources})
 endif()

 # Validation DLRM tests
 if (WMMA_BUILD_VALIDATION_TESTS)
     add_dlrm_validation_test(dlrm_dot_test-validate ${DlrmDotTestSources})
     add_dlrm_validation_test(dlrm_dot_lds_test-validate ${DlrmDotLdsTestSources})
 endif()
