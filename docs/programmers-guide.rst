.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _programmers-guide:

===================
Programmer's guide
===================

This document provides insight into the library source code organization, design implementation details, helpful information for new development, and testing and benchmarking details.

From a high level:

- ``Doxygen/Breathe/Sphinx/ReadTheDocs`` are used to produce documentation. The API documentation is generated using:

  - Doxygen comments in include files in the directory ``library/include``
  - files in the directory ``docs/source``.

- Jenkins is used to automate Continuous Integration (CI) testing (``.jenkins`` folder has configurations).

- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.

--------------------------------
Library source code organization
--------------------------------

The rocWMMA code is split into four major parts:

- The ``library`` directory contains the header library API and implementation.
- The ``samples`` directory contains real-world sample use-cases of the rocWMMA API.
- The ``test`` directory contains testing infrastructure for rocWMMA.
- The ``docs`` directory contains documentation generation sources.

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following structure:

- ``library/include/rocwmma/``: C++ include files for the rocWMMA API. These files also contain Doxygen content that documents the API.
The API currently has three API contexts:

  - ``rocwmma.hpp``: The main API for rocWMMA, defining fragment data abstractions, wave-wise storing, loading, matrix multiply-accumulate (mma)
  and threadblock synchronization. This API's function signatures are portable from nvcuda::wmma.
  - ``rocwmma_coop.hpp``: A complimentary API for rocWMMA, defining functionality that allows GPU wavefronts to collaborate in the loading / storing
  of fragment data. These are unique to rocWMMA.
  - ``rocwmma_transforms.hpp``: A complimentary API for rocWMMA, defining functionality to manipulate fragment data (e.g. transpose and data layout changes).
  These are unique to rocWMMA.

- ``library/include/internal``: Internal include files define the main infrastructure driving the rocWMMA API:

  - Configuration of platforms and architectures
  - Type support
  - Input and output configuration, shapes and traits
  - Loading and storing utilities
  - Layouts of memory and registers
  - Mapping utilities
  - Intrinsic wrappers
  - Vector class implementations
  - Vector conversion, permutation and transform utilities
  - Vector packing and unpacking
  - Matrix multiply-accumulate
  - Cooperative loading and storing
  - Threadblock synchronization and flow control
  - Utility code

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following use cases:

- ``samples/hipRTC_gemm.cpp``: For calling simple General Matrix Multiply (GEMM) algorithm demonstration without LDS memory usage and no transpose, from within the hipRTC environment

- ``samples/simple_sgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for single-precision floating point types

- ``samples/simple_dgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for double-precision floating point types

- ``samples/simple_sgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for single-precision floating point types

- ``samples/simple_dgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for double-precision floating point types

- ``samples/simple_hgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for half-precision floating point types

- ``samples/perf_sgemm.cpp``: For calling the high performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for single-precision floating point types

- ``samples/perf_dgemm.cpp``: For calling the high performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for double-precision floating point types

- ``samples/perf_hgemm.cpp``: For calling the high performant multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for half-precision floating point types

- ``samples/simple_dlrm.cpp``: For calling simple Deep Learning Recommendation Model (DLRM) for machine learning

- ``samples/common.hpp``: Common code used by all the above rocWMMA samples files

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test code support:

- ``test/bin``: To generate benchmark plots from the ``gtest`` output dumps of rocWMMA's benchmark tests.

- ``test/device``: Device utility kernels to support test setup and validation on GPU.

- ``test/dlrm``: For various strategies of DLRM application. This test is used to validate DLRM functions using rocWMMA API.

- ``test/gemm``: For various strategies of GEMM application. This test is used to validate and benchmark GEMM functions using rocWMMA API.

- ``test/unit``: For testing the basic functional units of rocWMMA library.

``docs`` directory
^^^^^^^^^^^^^^^^^^^

- Sphinx and Doxygen are used to generate project documentation.
- ``api-reference-guide.rst`` pulls from Doxygen documentation to format API documentation.
- ``installation.rst`` builds installation / build instructions for rocWMMA.
- ``license.rst`` includes information pertaining to rocWMMA licensing.
- ``programmers-guide.rst`` includes information about project organization and expectations.
- ``what-is-rocwmma.rst`` includes a description of rocWMMA.

Contributing
^^^^^^^^^^^^

For those wishing to contribute to the project, please see `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.
