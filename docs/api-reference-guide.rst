.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _api-reference-guide:

********************
API reference guide
********************

This document provides information about rocWMMA functions, data types, and other programming constructs.

Synchronous API
^^^^^^^^^^^^^^^

In general, rocWMMA API functions ( ``load_matrix_sync``, ``store_matrix_sync``, ``mma_sync`` ) are assumed to be synchronous when
used in the context of global memory.

When using these functions in the context of shared memory (e.g. LDS memory), additional explicit workgroup synchronization (``synchronize_workgroup``)
may be required due to the nature of this memory usage.


Supported architectures
^^^^^^^^^^^^^^^^^^^^^^^^

List of supported CDNA architectures (wave64):

* gfx908
* gfx90a
* gfx940
* gfx941
* gfx942

.. note::
    gfx9 = gfx908, gfx90a, gfx940, gfx941, gfx942

    gfx940+ = gfx940, gfx941, gfx942


List of supported RDNA architectures (wave32):

* gfx1100
* gfx1101
* gfx1102

.. note::
    gfx11 = gfx1100, gfx1101, gfx1102


Supported data types
^^^^^^^^^^^^^^^^^^^^

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>, where:

* Input Type = Matrix A / B
* Output Type = Matrix C / D
* Compute Type = Math / accumulation type

.. tabularcolumns::
   |C|C|C|C|

+------------------------------+------------+-----------+---------------+----------------------------+
|Ti / To / Tc                  |BlockM      |BlockN     |BlockK Range*  |   Architecture Support     |
|                              |            |           |(Powers of 2)  |                            |
+==============================+============+===========+===============+============================+
|                              |16          |16         | 32+           |                            |
|     bf8 / f32 / f32          +------------+-----------+---------------+          gfx940+           |
|                              |32          |32         | 16+           |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 32+           |                            |
|     f8 / f32 / f32           +------------+-----------+---------------+          gfx940+           |
|                              |32          |32         | 16+           |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |            |           | 16+           |   gfx908, gfx90a, gfx11    |
|                              |     16     |    16     +---------------+----------------------------+
|                              |            |           | 32+           |          gfx940+           |
|     i8 / i32 / i32           +------------+-----------+---------------+----------------------------+
|                              |            |           | 8+            |      gfx908, gfx90a        |
|                              |     32     |    32     +---------------+----------------------------+
|                              |            |           | 16+           |          gfx940+           |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |            |           | 16+           |   gfx908, gfx90a, gfx11    |
|                              |     16     |    16     +---------------+----------------------------+
|                              |            |           | 32+           |          gfx940+           |
|     i8 / i8 / i32            +------------+-----------+---------------+----------------------------+
|                              |            |           | 8+            |      gfx908, gfx90a        |
|                              |     32     |    32     +---------------+----------------------------+
|                              |            |           | 16+           |          gfx940+           |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 16+           |        gfx9, gfx11         |
|     f16 / f32 / f32          +------------+-----------+---------------+----------------------------+
|                              |32          |32         | 8+            |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 16+           |        gfx9, gfx11         |
|     f16 / f16 / f32          +------------+-----------+---------------+----------------------------+
|                              |32          |32         | 8+            |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 16+           |        gfx9, gfx11         |
|     f16 / f16 / f16**        +------------+-----------+---------------+----------------------------+
|                              |32          |32         | 8+            |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |            |           | 8+            |          gfx908            |
|                              |     16     |    16     +---------------+----------------------------+
|                              |            |           | 16+           |    gfx90a, gfx940+, gfx11  |
|     bf16 / f32 / f32         +------------+-----------+---------------+----------------------------+
|                              |            |           | 4+            |          gfx908            |
|                              |     32     |    32     +---------------+----------------------------+
|                              |            |           | 8+            |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |            |           | 8+            |          gfx908            |
|                              |     16     |    16     +---------------+----------------------------+
|                              |            |           | 16+           |    gfx90a, gfx940+, gfx11  |
|     bf16 / bf16 / f32        +------------+-----------+---------------+----------------------------+
|                              |            |           | 4+            |          gfx908            |
|                              |     32     |    32     +---------------+----------------------------+
|                              |            |           | 8+            |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |            |           | 8+            |          gfx908            |
|                              |     16     |    16     +---------------+----------------------------+
|                              |            |           | 16+           |    gfx90a, gfx940+, gfx11  |
|     bf16 / bf16 / bf16**     +------------+-----------+---------------+----------------------------+
|                              |            |           | 4+            |          gfx908            |
|                              |     32     |    32     +---------------+----------------------------+
|                              |            |           | 8+            |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 4+            |           gfx9             |
|     f32 / f32 / f32          +------------+-----------+---------------+----------------------------+
|                              |32          |32         | 2+            |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|                              |16          |16         | 8+            |                            |
|     xf32 / xf32 / xf32       +------------+-----------+---------------+          gfx940+           |
|                              |32          |32         | 4+            |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|      f64 / f64 / f64         |16          |16         | 4+            |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+

.. note::
    \* = BlockK range lists the minimum possible value. Other values in the range are powers of 2 larger than the minimum. Practical BlockK values aren't usually larger than 64.
    \*\* = CDNA architectures matrix unit accumulation is natively 32-bit precision and is converted to the desired type.



**= f64 datatype is only supported on MI-200 class AMDGPU and successors.

Supported matrix layouts
^^^^^^^^^^^^^^^^^^^^^^^^

(N = col major, T = row major)

.. tabularcolumns::
   |C|C|C|C|

+---------+--------+---------+--------+
|LayoutA  |LayoutB |Layout C |LayoutD |
+=========+========+=========+========+
|N        |N       |N        |N       |
+---------+--------+---------+--------+
|N        |N       |T        |T       |
+---------+--------+---------+--------+
|N        |T       |N        |N       |
+---------+--------+---------+--------+
|N        |T       |T        |T       |
+---------+--------+---------+--------+
|T        |N       |N        |N       |
+---------+--------+---------+--------+
|T        |N       |T        |T       |
+---------+--------+---------+--------+
|T        |T       |N        |N       |
+---------+--------+---------+--------+
|T        |T       |T        |T       |
+---------+--------+---------+--------+

-----------------
Using rocWMMA API
-----------------

This section describes how to use the rocWMMA library API.

rocWMMA datatypes
^^^^^^^^^^^^^^^^^

matrix_a
''''''''

.. doxygenstruct:: rocwmma::matrix_a


matrix_b
''''''''

.. doxygenstruct:: rocwmma::matrix_b


accumulator
'''''''''''

.. doxygenstruct:: rocwmma::accumulator


row_major
'''''''''

.. doxygenstruct:: rocwmma::row_major


col_major
'''''''''

.. doxygenstruct:: rocwmma::col_major


VecT
''''

.. doxygenclass:: VecT



IOConfig
''''''''''''

.. doxygenstruct:: rocwmma::IOConfig


IOShape
''''''''''''

.. doxygenstruct:: rocwmma::IOShape

rocWMMA enumeration
^^^^^^^^^^^^^^^^^^^

.. note::
    The enumeration constants numbering is consistent with the standard C++ libraries.

layout_t
''''''''''''

.. doxygenenum:: rocwmma::layout_t


rocWMMA API functions
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: fill_fragment

.. doxygenfunction:: load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag, const DataT* data, uint32_t ldm)

.. doxygenfunction:: load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag, const DataT* data, uint32_t ldm, layout_t layout)

.. doxygenfunction:: store_matrix_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag, uint32_t ldm)

.. doxygenfunction:: store_matrix_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag, uint32_t ldm,layout_t layout)

.. doxygenfunction:: mma_sync

.. doxygenfunction:: synchronize_workgroup

.. doxygenfunction:: load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag, const DataT* data, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount, uint32_t splitCount)

.. doxygenfunction:: load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag, const DataT* data, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)

.. doxygenfunction:: load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout>& frag, const DataT* data, uint32_t ldm)

.. doxygenfunction:: store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount, uint32_t splitCount)

.. doxygenfunction:: store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)

.. doxygenfunction:: store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayout> const& frag, uint32_t ldm)

Sample programs
^^^^^^^^^^^^^^^^

See a sample code for calling rocWMMA functions ``load_matrix_sync``, ``store_matrix_sync``, ``fill_fragment``, and ``mma_sync`` `here <https://github.com/ROCm/rocWMMA/blob/develop/samples/simple_hgemm.cpp>`_.
For more such sample programs, refer to the `Samples directory <https://github.com/ROCm/rocWMMA/tree/develop/samples>`_.
