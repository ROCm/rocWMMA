===============================
Getting Started Guide for Linux
===============================

------------
Introduction
------------

This document contains instructions for installing, using, and contributing to rocWMMA.
The quickest way to install is from prebuilt packages. Alternatively, there are instructions to build from source. The document also contains an API Reference Guide, Programmer's Guide, and Contributor's Guides.

Documentation Roadmap
^^^^^^^^^^^^^^^^^^^^^
The following is a list of rocWMMA documents in the suggested reading order:

 - Getting Started Guide (this document): Describes how to install and configure the rocWMMA library; designed to get users up and running quickly with the library.
 - API Reference Guide : Provides detailed information about rocWMMA functions, data types and other programming constructs.
 - Programmer's Guide: Describes the code organization, Design implementation detail, Optimizations used in the library and those that should be considered for new development and Testing & Benchmarking detail.
 - Contributor's Guide : Describes coding guidelines for contributors.

-------------
Prerequisites
-------------

-  A ROCm enabled platform, more information `here <https://rocm.github.io/>`_.


-----------------------------
Installing pre-built packages
-----------------------------

rocWMMA can be installed on Ubuntu or Debian using

::

   sudo apt-get update
   sudo apt-get install rocWMMA

rocWMMA can be installed on CentOS using

::

    sudo yum update
    sudo yum install rocWMMA

rocWMMA can be installed on SLES using

::

    sudo dnf upgrade
    sudo dnf install rocWMMA

Once installed, rocWMMA can be used just like any other library with a C++ API.
The rocwmma.hpp header file will need to be included in the user code in order to make calls
into rocWMMA.

Once installed, rocwmma.hpp can be found in the /opt/rocm/include directory.
Only this installed file should be used when needed in user code.
Other rocWMMA files can be found in /opt/rocm/include/internal, however these files
should not be directly included.


-------------------------------
Building and Installing rocWMMA
-------------------------------

For most users building from source is not necessary, as rocWMMA can be used after installing the pre-built
packages as described above. If desired, the following instructions can be used to build rocWMMA from source.

Requirements
^^^^^^^^^^^^
As a general rule, 5GB of system memory is required for a full rocWMMA build. This value can be lower if rocWMMA is built without tests. This value may also increase in the future as more functions are added to rocWMMA.


Download rocWMMA
^^^^^^^^^^^^^^^^

The rocWMMA source code is available at the `rocWMMA github page <https://github.com/ROCmSoftwarePlatform/rocWMMA>`_. rocWMMA has a minimum ROCm support version 4.3.
Check the ROCm Version on your system. For Ubuntu use

::

    apt show rocm-libs -a

For Centos use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example the ROCm version could be 4.0.0.40000-23, this corresponds to major = 4, minor = 0, patch = 0, build identifier 40000-23.
There are GitHub branches at the rocWMMA site with names rocm-major.minor.x where major and minor are the same as in the ROCm version. For ROCm version 4.0.0.40000-23 you need to use the following to download rocWMMA:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocWMMA.git
   cd rocWMMA

Replace x.y in the above command with the version of ROCm installed on your machine. For example: if you have ROCm 5.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-5.0

The user can build either

* library

* library + samples

* library + tests

* library + tests + assembly

You only need (library) if you call rocWMMA from your code.
The client contains the test samples and benchmark code.

Below are the project options available to build rocWMMA library with/without clients.

.. tabularcolumns::
   |C|C|C|

+------------------------------+-------------------------------------+-----------------------------------------------+
|Option                        |Description                          |Default Value                                  |
+==============================+=====================================+===============================================+
|AMDGPU_TARGETS                |Build code for specific GPU target(s)|	gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+    |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_TESTS           |Build Tests                          |ON                                             |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_SAMPLES         |Build Samples                        |ON                                             |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_ASSEMBLY        |Generate assembly files              |OFF                                            |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_VALIDATION_TESTS|Build validation tests               |ON (requires ROCWMMA_BUILD_TESTS=ON)           |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_BENCHMARK_TESTS |Build benchmark tests                |OFF (requires ROCWMMA_BUILD_TESTS=ON)          |
+------------------------------+-------------------------------------+-----------------------------------------------+
|ROCWMMA_BUILD_EXTENDED_TESTS  |Build extended testing coverage      |OFF (requires ROCWMMA_BUILD_TESTS=ON)          |
+------------------------------+-------------------------------------+-----------------------------------------------+
|WMMA_VALIDATE_WITH_ROCBLAS    |Use rocBLAS for validation tests     |ON (requires ROCWMMA_BUILD_VALIDATION_TESTS=ON)|
+------------------------------+-------------------------------------+-----------------------------------------------+
|WMMA_BENCHMARK_WITH_ROCBLAS   |Include rocBLAS benchmarking data    |OFF (requires ROCWMMA_BUILD_BENCHMARK_TESTS=ON)|
+------------------------------+-------------------------------------+-----------------------------------------------+


Build only library
^^^^^^^^^^^^^^^^^^

CMake has a minimum version requirement 3.5.

Minimum ROCm version support is 4.3.

By default, the project is configured as Release mode.

To build only library, run the following comomand :

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF

Here are some other example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|         Configuration             |                                          Command                                                                   |
+===================================+====================================================================================================================+
|            Basic                  |                                CC=hipcc CXX=hipcc cmake -B<build_dir> .                                            |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|        Targeting MI100            |                   CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-                          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|          Debug build              |                    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug                               |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
| Build without rocBLAS(default on) |  CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

After configuration, build with

    cmake --build <build_dir> -- -j


Build library + samples
^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run the following comomand :

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=ON

After configuration, build with

    cmake --build <build_dir> -- -j

The samples folder in <build_dir> contains executables in the table below.

================ ===========================================================================
executable name  description
================ ===========================================================================
simple-gemm      a simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API
sgemv            a simple GEMV operation [y = alpha * (A) * x + beta * y] using rocWMMA API
simple-dlrm      a simple DLRM operation using rocWMMA API
================ ===========================================================================


Build library + tests
^^^^^^^^^^^^^^^^^^^^^
rocWMMA has several test suites that can be built:

- DLRM tests
- GEMM tests
- Unit tests

DLRM tests cover the dot product interactions between embeddings used in DLRM.

GEMM tests cover block-wise Generalized Matrix Multiplication (GEMM) implemented with rocWMMA.

Unit tests cover various aspects of rocWMMA API and internal functionality.

rocWMMA can build both validation and benchmark tests. The library uses CPU or rocBLAS methods for validation (where available) and benchmark comparisons based on the provided project option.
By default, the project is linked against rocBLAS for validating results. 
Minimum ROCBLAS library version requirement is 4.0.

To build library and tests, run the following command :

    CC=hipcc CXX=hipcc cmake -B<build_dir> .

After configuration, build with

    cmake --build <build_dir> -- -j

The tests in <build_dir> contains executables in the table below.

====================================== ===========================================================================================================
executable name                        description
====================================== ===========================================================================================================
dlrm/dlrm_dot_test-*                   a DLRM implementation using rocWMMA API
dlrm/dlrm_dot_lds_test-*               a DLRM implementation using rocWMMA API with LDS shared memory
gemm/mma_sync_test-*                   a simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API 
gemm/mma_sync_multi_test-*             a modified GEMM operation, each wave targets a sub-grid of output blocks using rocWMMA API
gemm/mma_sync_multi_lds_test-*         a modified GEMM operation, each wave targets a sub-grid of output blocks using LDS memory and rocWMMA API
gemm/mma_sync_barrier_test-*           a simple GEMM operation with wave synchronization
unit/fill_fragment_test                tests fill_fragment API function
unit/load_store_matrix_sync_test       tests load_matrix_sync and store_matrix_sync API functions
unit/load_store_matrix_coop_sync_test  tests load_matrix_coop_sync and store_matrix_coop_sync API functions
unit/contamination_test                tests against contamination of pristine data for loads and stores
unit/layout_test                       tests accuracy of internal matrix layout patterns
unit/mapping_util_test                 tests mapping utilities used in rocWMMA implementations
unit/vector_iterator_test              tests internal vector storage implementation
====================================== ===========================================================================================================

*= validate: executables that compare outputs for correctness against reference sources such as CPU or rocBLAS calculations.

*= bench: executables that measure kernel execution speeds and may compare against those of rocBLAS references.


Build library + Tests + Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and tests with assembly code generation, run the following command :

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_ASSEMBLY=ON

After configuration, build with

    cmake --build <build_dir> -- -j

The assembly folder in <build_dir> contains assembly generation of test executables in the format [test_executable_name.s]
