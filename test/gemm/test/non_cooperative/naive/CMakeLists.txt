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

# Cooperative workgroup test suite
set(GemmNaiveTestSources ${GemmCommonSources}
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_16x16_nn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_16x16_nt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_16x16_tn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_16x16_tt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_32x32_nn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_32x32_nt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_32x32_tn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_test_32x32_tt.cpp
                          )

# Ad hoc test
set(GemmNaiveAdHocTestSources ${GemmCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_naive_ad_hoc_test.cpp)

# Validation and bench tests
add_gemm_test(gemm_non_coop_naive_test ${GemmNaiveTestSources})
add_gemm_test(gemm_non_coop_naive_ad_hoc_test ${GemmNaiveAdHocTestSources})
