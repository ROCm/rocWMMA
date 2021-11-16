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

set(WMMA_TEST_UNIT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Unit common test sources
set(UnitCommonSources ${WMMA_COMMON_TEST_SOURCES}
                       UnitKernelBase.cpp)

set(ContaminationTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContamination_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContamination_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContamination_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContamination_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContamination_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContamination_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContamination_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContamination_32x32_T.cpp)

set(LayoutTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColNTLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColNTLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColNTLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColNTLayout_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowNTLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowNTLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowNTLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowNTLayout_32x32_T.cpp)

set(MapUtilTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapBlockToMatrix_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapBlockToMatrix_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapBlockToMatrix_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapBlockToMatrix_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapMatrixToData_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapMatrixToData_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapMatrixToData_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapMatrixToData_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapThreadToMatrix_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapThreadToMatrix_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapThreadToMatrix_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapThreadToMatrix_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapWaveToMatrix_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapWaveToMatrix_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapWaveToMatrix_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/MapUtilTest/MapWaveToMatrix_32x32_T.cpp)

set(LoadStoreMatrixSyncTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSync_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSync_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSync_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSync_32x32_T.cpp)

set(FillFragmentTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/FillFragmentTest/FillFragment_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/FillFragmentTest/FillFragment_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/FillFragmentTest/FillFragment_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/FillFragmentTest/FillFragment_32x32_T.cpp)

set(VectorIteratorTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/VectorIteratorTest/VectorIterator.cpp)

# Benchmark Unit
add_wmma_test(ContaminationTest ${ContaminationTestSources})
add_wmma_test(LayoutTest ${LayoutTestSources})
add_wmma_test(MapUtilTest ${MapUtilTestSources})
add_wmma_test(LoadStoreMatrixSyncTest ${LoadStoreMatrixSyncTestSources})
add_wmma_test(FillFragmentTest ${FillFragmentTestSources})
add_wmma_test(VectorIteratorTest ${VectorIteratorTestSources})

