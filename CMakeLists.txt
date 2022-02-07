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

cmake_minimum_required( VERSION 3.5 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()
message( VERBOSE "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set( CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

### Project WMMA
project( WMMA LANGUAGES CXX )
set (CMAKE_CXX_STANDARD 14)

# Top level configs
if( CMAKE_PROJECT_NAME STREQUAL "WMMA" )
  option( WMMA_BUILD_TESTS "Build WMMA tests" ON )
  option( WMMA_BUILD_SAMPLES "Build WMMA samples" ON )
  option( WMMA_BUILD_DOCS "Build doxygen documentation" OFF )
  option( WMMA_BUILD_ASSEMBLY "Output assembly files" OFF )
endif()

# set( AMDGPU_TARGETS "gfx908:xnack-" ) # User variable
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif()
if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix for packages.")
  set(CPACK_SET_DESTDIR OFF)
endif()

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

find_package(ROCM 0.7 CONFIG REQUIRED)
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)
include(ROCMClients)

rocm_setup_version(VERSION "0.7")

rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
  TARGETS "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+" )

# Variable AMDGPU_TARGET must be a cached variable and must be specified before calling find_package(hip)
# This is because hip-config.cmake sets --offload-arch via AMDGPU_TARGET cached variable __after__ setting
# default cached variable AMDGPU_TARGET to "gfx900;gfx906;gfx908", where not all archs are compatible with MFMA instructions
#
# By rule, once cached variable is set, it cannot be overridden unless we use the FORCE option
if(AMDGPU_TARGETS)
  set(AMDGPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message( VERBOSE "AMDGPU_TARGETS=${AMDGPU_TARGETS}")

find_package( hip REQUIRED )
find_package( OpenMP REQUIRED )

add_library(wmma INTERFACE)
target_link_libraries(wmma INTERFACE hip::device hip::host OpenMP::OpenMP_CXX)
target_compile_options(wmma INTERFACE -std=c++14)

rocm_install_targets(
  TARGETS wmma
  INCLUDE library/include
)

if(WMMA_BUILD_DOCS)
  add_subdirectory(docs)
endif()

if(WMMA_BUILD_SAMPLES)
  rocm_package_setup_client_component(
    samples
  )
  add_subdirectory(samples)
endif()


if(WMMA_BUILD_TESTS)
  rocm_package_setup_client_component(
    tests
    DEPENDS
      COMMON "rocblas >= 2.32.0"
  )
  add_subdirectory(test)
endif()

if(WMMA_BUILD_SAMPLES OR WMMA_BUILD_TESTS)
  rocm_package_setup_component(clients)
endif()

# Package
rocm_package_add_dependencies("hip-rocclr >= 3.5.0")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
  NAME WMMA
  DESCRIPTION "AMD's C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications on GPU using MFMA hardware cores."
  MAINTAINER "WMMA Maintainer <wmma-maintainer@amd.com>"
  HEADER_ONLY
)