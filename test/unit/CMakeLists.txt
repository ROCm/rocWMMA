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

set(WMMA_TEST_UNIT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Unit common test sources
set(UnitCommonSources ${WMMA_COMMON_TEST_SOURCES}
                       UnitKernelBase.cpp)

set(ContaminationTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationA_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationA_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationA_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationA_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationA_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationB_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationB_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationB_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationB_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationB_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationAcc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationAcc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationAcc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationAcc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/LoadContaminationAcc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationA_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationA_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationA_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationA_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationA_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationAcc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationAcc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationAcc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationAcc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationAcc_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationB_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationB_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationB_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationB_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/ContaminationTest/StoreContaminationB_256.cpp
                     )

set(LayoutTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/ColLayout_32x32_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_16x16_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_16x16_T.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_32x32_N.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LayoutTest/RowLayout_32x32_T.cpp)

#set(LdsMappingTestSources ${UnitCommonSources}
#                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_16_N_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_16_T_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_32_N_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_32_T_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_16_N_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_16_T_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_32_N_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_32_T_LdsKW.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_16_N_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_16_T_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_32_N_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_32_T_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_16_N_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_16_T_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_32_N_LdsKH.cpp
                     #  ${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingB_32_T_LdsKH.cpp
                     #${CMAKE_CURRENT_SOURCE_DIR}/test/LdsMappingTest/LdsMappingA_16_N_LdsRF.cpp
#)

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
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncA_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncA_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncA_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncA_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncA_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncB_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncB_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncB_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncB_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncB_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncAcc_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncAcc_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncAcc_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncAcc_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixSyncTest/LoadStoreMatrixSyncAcc_256.cpp
                    )

set(LoadStoreMatrixCoopSyncTestSources ${UnitCommonSources}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncA_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncA_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncA_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncA_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncA_256.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncAcc_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncAcc_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncAcc_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncAcc_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncAcc_256.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncB_16.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncB_32.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncB_64.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncB_128.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/LoadStoreMatrixCoopSyncTest/LoadStoreMatrixCoopSyncB_256.cpp
                 )

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
#add_wmma_test(LdsMappingTest ${LdsMappingTestSources})
add_wmma_test(MapUtilTest ${MapUtilTestSources})
add_wmma_test(LoadStoreMatrixSyncTest ${LoadStoreMatrixSyncTestSources})
add_wmma_test(LoadStoreMatrixCoopSyncTest ${LoadStoreMatrixCoopSyncTestSources})
add_wmma_test(FillFragmentTest ${FillFragmentTestSources})
add_wmma_test(VectorIteratorTest ${VectorIteratorTestSources})

