.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _programmers-guide:

===================
Programmer's guide
===================

This document provides insight into the library source code organization, design implementation details, helpful information for new development, and testing and benchmarking details.

--------------------------------
Library source code organization
--------------------------------

The rocWMMA code is split into four major parts:

- The ``library`` directory contains the library source code.
- The ``samples`` directory contains real-world use-cases of the rocWMMA API.
- The ``test`` directory contains validation tests for rocWMMA API.
- Infrastructure

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following include files:

- ``library/include/rocwmma/``: C++ include files for the hipTensor API. These files also contain Doxygen comments that document the API.

- ``library/include/internal``: Internal include files for:

  - Type support
  - Input and output configuration, shapes and traits
  - Layout
  - Mapping Utility
  - Cross-lane operation utility
  - Vector blend utility
  - Packing and unpacking
  - Conversion and broadcasting
  - Load and store
  - Matrix multiply-accumulate
  - Cooperative load and store
  - Threadblock synchronization
  - Utility code

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following use cases:

- ``samples/hipRTC_gemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose, from within the hipRTC environment

- ``samples/simple_sgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for single-precision floating point types

- ``samples/simple_dgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for double-precision floating point types

- ``samples/simple_sgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for single-precision floating point types

- ``samples/simple_dgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for double-precision floating point types

- ``samples/simple_hgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for half-precision floating point types

- ``samples/perf_sgemm.cpp``: For calling the best performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and
optimized pipeline for single-precision floating point types

- ``samples/perf_dgemm.cpp``: For calling the best performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and
optimized pipeline for double-precision floating point types

- ``samples/perf_hgemm.cpp``: For calling the best performant multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and
optimized pipeline for half-precision floating point types

- ``samples/simple_dlrm.cpp``: For calling simple Deep Learning Recommendation Model (DLRM) for machine learning

- ``samples/common.hpp``: Common code used by all the above rocWMMA samples files

The `test` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test codes for testing the following functionalities:

- ``test/bin``: To generate benchmark plots from the ``gtest`` output dumps of rocWMMA's benchmark tests.

- ``test/dlrm``: For various strategies of DLRM application. This test is used to validate DLRM functions using rocWMMA API.

- ``test/gemm``: For various strategies of GEMM application. This test is used to validate and benchmark GEMM functions using rocWMMA API.

- ``test/unit``: For testing the basic functional units of rocWMMA library.

Infrastructure
^^^^^^^^^^^^^^

- CMake is used to build and package rocWMMA. There are ``CMakeLists.txt`` files throughout the code.

- ``Doxygen/Breathe/Sphinx/ReadTheDocs`` are used to produce documentation. The API documentation is generated using:

  - Doxygen comments in include files in the directory ``library/include``
  - files in the directory ``docs/source``.

- Jenkins is used to automate Continuous Integration (CI) testing.

- ``clang-format`` is used to format C++ code.
