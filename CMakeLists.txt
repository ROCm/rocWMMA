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

cmake_minimum_required( VERSION 3.5 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()
message( VERBOSE "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

### Project WMMA
project( WMMA LANGUAGES CXX )

# set( AMDGPU_TARGETS "gfx908:xnack-" ) # User variable
set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if(COMMAND rocm_check_target_ids)
  rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
    TARGETS "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+" )
else()
  # Detect compiler support for target ID
  # This section is deprecated. Please use rocm_check_target_ids for future use.
  message( VERBOSE "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}" )
  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
      OUTPUT_VARIABLE CXX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
  endif()
  if(TARGET_ID_SUPPORT)
    message( VERBOSE "target id support")
    set(DEFAULT_AMDGPU_TARGETS "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+")
  else()
    message( VERBOSE "NO target id support")
    set(DEFAULT_AMDGPU_TARGETS "gfx908;gfx90a")
  endif()
endif()

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

set(WMMA_VALIDATE_WITH_ROCBLAS ON CACHE STRING "Use rocBLAS for validation")
if(WMMA_VALIDATE_WITH_ROCBLAS)
  find_package( rocblas REQUIRED PATHS /opt/rocm /opt/rocm/rocblas )
endif()

add_library(wmma INTERFACE)
target_include_directories(wmma INTERFACE ${PROJECT_SOURCE_DIR}/library/include)
target_link_libraries(wmma INTERFACE hip::device hip::host)
target_compile_options(wmma INTERFACE -std=c++14)

enable_testing()
add_subdirectory(test)
