==============
Installation
==============

This document provides instructions for installing rocWMMA.
The quickest way to install is using prebuilt packages. Alternatively, there are instructions to build from source.

-------------
Prerequisites
-------------

A ROCm enabled platform, more information `here <https://github.com/ROCm/ROCm>`_.

-----------------------------
Installing pre-built packages
-----------------------------

To install rocWMMA on Ubuntu or Debian, use:

::

   sudo apt-get update
   sudo apt-get install rocWMMA

To install rocWMMA on CentOS, use:

::

    sudo yum update
    sudo yum install rocWMMA

To install rocWMMA on SLES, use:

::

    sudo dnf upgrade
    sudo dnf install rocWMMA

Once installed, rocWMMA can be used just like any other library with a C++ API.

Once rocWMMA is installed, you can see the ``rocwmma.hpp`` header file in the ``/opt/rocm/include`` directory.
You must include only ``rocwmma.hpp`` in the user code to make calls into rocWMMA. Don't directly include other rocWMMA files that are found in ``/opt/rocm/include/internal``.

-------------------------------
Building and Installing rocWMMA
-------------------------------

For most users building from source is not necessary, as rocWMMA can be used after installing the pre-built
packages as described above. If still desired, here are the instructions to build rocWMMA from source:

System Requirements
^^^^^^^^^^^^^^^^^^^
As a general rule, 8GB of system memory is required for a full rocWMMA build. This value can be lower if rocWMMA is built without tests. This value may also increase in the future as more functions are added to rocWMMA.


GPU Support
^^^^^^^^^^^
AMD CDNA class GPU featuring matrix core support: `gfx908`, `gfx90a` as `gfx9`

.. note::
    Double precision FP64 datatype support requires gfx90a

Or

AMD RDNA3 class GPU featuring AI acceleration support: `gfx1100`, `gfx1101`, `gfx1102` as `gfx11`

Download rocWMMA
^^^^^^^^^^^^^^^^^

The rocWMMA source code is available at the `rocWMMA github page <https://github.com/ROCmSoftwarePlatform/rocWMMA>`_. rocWMMA has a minimum ROCm support version 5.4.
To check the ROCm version on an Ubuntu system, use:

::

    apt show rocm-libs -a

On Centos, use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, a ROCm version 4.0.0.40000-23 corresponds to major = 4, minor = 0, patch = 0, and build identifier 40000-23.
There are GitHub branches at the rocWMMA site with names ``rocm-major.minor.x`` where major and minor are the same as in the ROCm version. To download rocWMMA on ROCm version 4.0.0.40000-23, use:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocWMMA.git
   cd rocWMMA

Replace ``x.y`` in the above command with the version of ROCm installed on your machine. For example, if you have ROCm 5.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-5.0.

You can choose to build any of the following:

* library

* library and samples

* library and tests

* library, tests, and assembly

You only need (library) for calling hipTensor from your code.
The client contains the test samples and benchmark code.

Below are the project options available to build rocWMMA library with or without clients.

.. list-table::

    *   -   **Option**
        -   **Description**
        -   **Default Value**
    *   -   AMDGPU_TARGETS
        -   Build code for specific GPU target(s)
        -   ``gfx908:xnack-``; ``gfx90a:xnack-``; ``gfx90a:xnack+``; ``gfx940``; ``gfx941``; ``gfx942``; ``gfx1100``; ``gfx1101``; ``gfx1102``
    *   -   ROCWMMA_BUILD_TESTS
        -   Build Tests
        -   ON
    *   -   ROCWMMA_BUILD_SAMPLES
        -   Build Samples
        -   ON
    *   -   ROCWMMA_BUILD_ASSEMBLY
        -   Generate assembly files
        -   OFF
    *   -   ROCWMMA_BUILD_VALIDATION_TESTS
        -   Build validation tests
        -   ON (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_BUILD_BENCHMARK_TESTS
        -   Build benchmark tests
        -   OFF (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_BUILD_EXTENDED_TESTS
        -   Build extended testing coverage
        -   OFF (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_VALIDATE_WITH_ROCBLAS
        -   Use rocBLAS for validation tests
        -   ON (requires ROCWMMA_BUILD_VALIDATION_TESTS=ON)
    *   -   ROCWMMA_BENCHMARK_WITH_ROCBLAS
        -   Include rocBLAS benchmarking data
        -   OFF (requires ROCWMMA_BUILD_BENCHMARK_TESTS=ON)

Build library
^^^^^^^^^^^^^^^^^^

ROCm-cmake has a minimum version requirement of 0.8.0 for ROCm 5.3.

Minimum ROCm version support is 5.4.

By default, the project is configured in Release mode.

To build the library alone, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF

Here are some other example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|         Configuration             |                                          Command                                                                   |
+===================================+====================================================================================================================+
|            Basic                  |                                ``CC=hipcc CXX=hipcc cmake -B<build_dir>``                                          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|        Targeting gfx908           |                   ``CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-``                      |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|          Debug build              |                    ``CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug``                           |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
| Build without rocBLAS(default on) |  ``CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF`` |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

Build library and samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

The samples folder in ``<build_dir>`` contains executables as given in the table below.

================ ==============================================================================================================================
Executable Name  Description
================ ==============================================================================================================================
``simple_sgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for single-precision floating point types
``simple_dgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for double-precision floating point types
``simple_hgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for half-precision floating point types

``perf_sgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for single-precision floating point types
``perf_dgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for double-precision floating point types
``perf_hgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for half-precision floating point types

``simple_sgemv``      A simple GEMV operation [y = alpha * (A) * x + beta * y] using rocWMMA API for single-precision fp32 inputs and output
``simple_dgemv``      A simple GEMV operation [y = alpha * (A) * x + beta * y] using rocWMMA API for double-precision fp64 inputs and output

``simple-dlrm``       A simple DLRM operation using rocWMMA API

``hipRTC_gemm``       A simple GEMM operation [D = alpha * (A x B) + beta * C] demonstrating runtime compilation (hipRTC) compatibility
================ ==============================================================================================================================


Build library and tests
^^^^^^^^^^^^^^^^^^^^^^^^^
rocWMMA provides the following test suites:

- DLRM tests: Cover the dot product interactions between embeddings used in DLRM
- GEMM tests: Cover block-wise Generalized Matrix Multiplication (GEMM) implemented with rocWMMA
- Unit tests: Cover various aspects of rocWMMA API and internal functionality

rocWMMA can build both validation and benchmark tests. The library uses CPU or rocBLAS methods for validation (when available) and benchmark comparisons based on the provided project option.
By default, the project is linked against rocBLAS for validating results.
Minimum ROCBLAS library version requirement for ROCm 4.3.0 is 2.39.0. 

To build library and tests, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> .

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

The tests in ``<build_dir>`` contain executables as given in the table below.

====================================== ===========================================================================================================
Executable Name                        Description
====================================== ===========================================================================================================
``dlrm/dlrm_dot_test-``*                   A DLRM implementation using rocWMMA API
``dlrm/dlrm_dot_lds_test-``*               A DLRM implementation using rocWMMA API with LDS shared memory
``gemm/mma_sync_test-``*                   A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API
``gemm/mma_sync_multi_test-``*             A modified GEMM operation where each wave targets a sub-grid of output blocks using rocWMMA API
``gemm/mma_sync_multi_ad_hoc_test-``*      An adhoc version of ``mma_sync_multi_test-``*
``gemm/mma_sync_multi_lds_test-``*         A modified GEMM operation where each wave targets a sub-grid of output blocks using LDS memory, rocWMMA API, and wave-level collaboration
``gemm/mma_sync_multi_lds_ad_hoc_test-``*  An adhoc version of ``mma_sync_multi_lds_test-``*
``gemm/mma_sync_coop_wg_test-``*           A modified GEMM operation where each wave targets a sub-grid of output blocks using LDS memory, rocWMMA API, and workgroup-level collaboration
``gemm/mma_sync_coop_wg_ad_hoc_test-``*    An adhoc version of ``mma_sync_coop_wg_test-``*
``gemm/barrier_test-``*                    A simple GEMM operation with wave synchronization
``unit/contamination_test``                Tests against contamination of pristine data for loads and stores
``unit/cross_lane_ops_test``               Tests cross-lane vector operations
``unit/fill_fragment_test``                Tests fill_fragment API function
``unit/io_shape_test``                     Tests input and output shape meta data
``unit/io_traits_test``                    Tests input and output logistical meta data
``unit/layout_test``                       Tests accuracy of internal matrix layout patterns
``unit/load_store_matrix_sync_test``       Tests ``load_matrix_sync`` and ``store_matrix_sync`` API functions
``unit/load_store_matrix_coop_sync_test``  Tests ``load_matrix_coop_sync`` and ``store_matrix_coop_sync`` API functions
``unit/map_util_test``                     Tests mapping utilities used in rocWMMA implementations
``unit/vector_iterator_test``              Tests internal vector storage iteration implementation
``unit/vector_test``                       Tests internal vector storage implementation
====================================== ===========================================================================================================

*= Validate: Executables that compare outputs for correctness against reference sources such as CPU or rocBLAS calculations.

*= Bench: Executables that measure kernel execution speeds and may compare against those of rocBLAS references.

Build library, tests, and assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the library and tests with assembly code generation, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_BUILD_ASSEMBLY=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

The assembly folder in ``<build_dir>`` contains assembly generation of test executables in the format ``test_executable_name.s``
