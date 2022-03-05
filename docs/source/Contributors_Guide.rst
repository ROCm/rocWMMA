===================
Contributor's Guide
===================

License Agreement
=================

1. The code I am contributing is mine, and I have the right to license
   it.

2. By submitting a pull request for this project I am granting you a
   license to distribute said code under the MIT License for the
   project.

Pull-request guidelines
=======================


Our code contriubtion guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
The rocWMMA repository follows a workflow which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code. Pull requests should:

-  target the **develop** branch for integration
-  ensure code builds successfully.
-  do not break existing test cases
-  new functionality will only be merged with new unit tests
-  new unit tests should integrate within the existing googletest framework.
-  tests must have good code coverage
-  code must also have benchmark tests, and performance must approach
   the compute bound limit or memory bound limit.

StyleGuide
==========

This project follows the `CPP Core
guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`__,
with few modifications or additions noted below. All pull-requests
should in good faith attempt to follow the guidelines stated therein,
but we recognize that the content is lengthy. Below we list our primary
concerns when reviewing pull-requests.

Interface
---------

-  Library code should use C++14
-  Our minimum supported compiler is hipcc 4.4
-  Avoid CamelCase
-  This rule applies specifically to publicly visible APIs, but is also
   encouraged (not mandated) for internal code

Philosophy
----------

-  `P.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus>`__:
   Write in ISO Standard C++14 (especially to support windows, linux and
   macos plaforms )
-  `P.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time>`__:
   Prefer compile-time checking to run-time checking

Implementation
--------------

-  `SF.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix>`__:
   Use a ``.cpp`` suffix for code files and an ``.hpp`` suffix for
   interface files if your project doesn't already follow another
   convention
-  `SF.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency>`__:
   A ``.cpp`` file must include the ``.hpp`` file(s) that defines its
   interface
-  `SF.7 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive>`__:
   Don't put a global ``using``-directive in a header file
-  `SF.8 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards>`__:
   Use ``#include`` guards for all ``.hpp`` files
-  `SF.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed>`__:
   Don't use an unnamed (anonymous) ``namespace`` in a header
-  `SL.10 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays>`__:
   Prefer using ``std::array`` or ``std::vector`` instead of a C array
-  `C.9 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private>`__:
   Minimize the exposure of class members
-  `F.3 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single>`__:
   Keep functions short and simple
-  `F.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi>`__:
   To return multiple 'out' values, prefer returning a ``std::tuple``
-  `R.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii>`__:
   Manage resources automatically using RAII (this includes
   ``std::unique_ptr`` & ``std::shared_ptr``)
-  `ES.11 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto>`__:
   Use ``auto`` to avoid redundant repetition of type names
-  `ES.20 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always>`__:
   Always initialize an object
-  `ES.23 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list>`__:
   Prefer the ``{}`` initializer syntax
-  `CP.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency>`__:
   Assume that your code will run as part of a multi-threaded program
-  `I.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`__:
   Avoid global variables

Format
------

C++ code is formatted using ``clang-format``. To run clang-format
use the version in the ``/opt/rocm/llvm/bin`` directory. Please do not use your
system's built-in ``clang-format``, as this may be an older version that
will result in different results.

To format a file, use:

::

    /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocWMMA directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install