
===================
Programmer's Guide
===================

--------------------------------
Library Source Code Organization
--------------------------------

The rocWMMA code is split into four major parts:

- The `library` directory contains all source code for the library.
- The `samples` directory contains real-world use-cases of the rocWMMA API.
- The `test` directory contains all validation, performance and unit tests of rocWMMA API.
- Infrastructure

The `library` directory
^^^^^^^^^^^^^^^^^^^^^^^

library/include/rocwmma/
''''''''''''''''''''''''

Contains C++ include files for the rocWMMA API. These files also contain Doxygen
comments that document the API.

library/include/internal
''''''''''''''''''''''''

Internal include files for:

- Type support
- Input / output configuration, shapes and traits
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


The `samples` directory
^^^^^^^^^^^^^^^^^^^^^^^
samples/hipRTC_gemm.cpp
''''''''''''''''''

sample code for calling Simple GEMM algorithm demonstration without LDS memory usage and no transpose, from within the hipRTC environment.

samples/sgemmv.cpp
''''''''''''''''''

sample code for calling Simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose.


samples/simple_gemm.cpp
'''''''''''''''''''''''

Sample code for calling Simple GEMM algorithm demonstration without LDS memory usage and no transpose.

samples/simple_dlrm.cpp
'''''''''''''''''''''''

Sample code for calling Simple Deep Learning Recommendation Model (DLRM) for machine learning.


samples/common.hpp
''''''''''''''''''

Common code used by all the above rocWMMA samples files.


The `test` directory
^^^^^^^^^^^^^^^^^^^^^^^

test/bin
''''''''

Script to generate benchmark plots from the gtest output dumps of benchmark tests of rocWMMA.

test/dlrm
'''''''''

Test code for various strategies of DLRM application. This test is used to validate dlrm functions using rocWMMA API.

test/gemm
'''''''''

Test Code for various strategies of GEMM application. This test is used to validate and benchmark GEMM functions using rocWMMA API.

test/unit
'''''''''

Test code for testing the basic functional units of rocWMMA library.


Infrastructure
^^^^^^^^^^^^^^

- CMake is used to build and package rocWMMA. There are CMakeLists.txt files throughout the code.
- Doxygen/Breathe/Sphinx/ReadTheDocs are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory library/include
  - files in the directory docs/source.

- Jenkins is used to automate Continuous Integration testing.
- clang-format is used to format C++ code.
