###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Add the current folder to test includes
set(ROCWMMA_TEST_GEMM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${ROCWMMA_TEST_GEMM_INCLUDE_DIRS})

# Setup kernel test symbols
set(ROCWMMA_KERNEL_BASE_NAME "gemm_PGR1_LB2_MP0_MB_CP")
set(ROCWMMA_TARGET_NAME ${ROCWMMA_KERNEL_BASE_NAME})
set(ROCWMMA_TARGET_SOURCES ${ROCWMMA_TARGET_NAME}_sources)

set(ROCWMMA_AD_HOC_TARGET_NAME ${ROCWMMA_TARGET_NAME}_ad_hoc)
set(ROCWMMA_AD_HOC_TARGET_SOURCES ${ROCWMMA_AD_HOC_TARGET_NAME}_sources)

# Populate with common sources to start
set(${ROCWMMA_TARGET_SOURCES} ${GemmCommonSources})

# Include all sources from testing contexts
add_subdirectory(test/block)
add_subdirectory(test/wave)
add_subdirectory(test/workgroup)

# Ad hoc test
# Note: GemmKernelBase and GemmResource instantiations required.
set(${ROCWMMA_AD_HOC_TARGET_SOURCES} ${ROCWMMA_COMMON_TEST_SOURCES}
                                     ${CMAKE_CURRENT_SOURCE_DIR}/test/ad_hoc_test.cpp)

add_gemm_test(${ROCWMMA_AD_HOC_TARGET_NAME} ${${ROCWMMA_AD_HOC_TARGET_SOURCES}})
