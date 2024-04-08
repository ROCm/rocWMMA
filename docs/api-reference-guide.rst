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

Supported data types
^^^^^^^^^^^^^^^^^^^^

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>

where,

Input Type = Matrix A/B

Output Type = Matrix C/D

Compute Type = Math / accumulation type

.. tabularcolumns::
   |C|C|C|C|

+------------------------------+------------+-----------+---------------+----------------------------+
|Ti / To / Tc                  |BlockM      |BlockN     |BlockK         |   Architecture Support     |
+==============================+============+===========+===============+============================+
|                              |16          |16         |Min: 32, pow2  |                            |
|     bf8 / f32 / f32          +------------+-----------+---------------+         gfx940+            |
|                              |32          |32         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f8 / f32 / f32                |16          |16         |Min: 32, pow2  |     gfx940+                |
+------------------------------+------------+-----------+---------------+----------------------------+
|f8 / f32 / f32                |32          |32         |Min: 16, pow2  |     gfx940+                |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i32 / i32                |16          |16         |Min: 16, pow2  |   gfx908, gfx90a, gfx11+   |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i32 / i32                |16          |16         |Min: 32, pow2  |     gfx940+                |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i32 / i32                |32          |32         |Min: 8, pow2   |   gfx908, gfx90a           |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i32 / i32                |32          |32         |Min: 16, pow2  |     gfx940+                |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i8 / i32                 |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|i8 / i32 / i32                |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f32 / f32               |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f32 / f32               |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f16 / f32               |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f16 / f32               |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f16 / f16*              |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f16 / f16 / f16*              |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / f32 / f32            |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / f32 / f32            |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / __half / f32         |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / __half / f32         |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / __half / __half*     |16          |16         |Min: 16, pow2  |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|__half / __half / __half*     |32          |32         |Min: 8, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / f32 / f32              |16          |16         |Min: 8, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / f32 / f32              |16          |16         |Min: 16, pow2  |    gfx90a, gfx940+, gfx11  |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / f32 / f32              |32          |32         |Min: 4, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / f32 / f32              |32          |32         |Min: 8, pow2   |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / f32             |16          |16         |Min: 8, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / f32             |16          |16         |Min: 16, pow2  |    gfx90a, gfx940+, gfx11  |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / f32             |32          |32         |Min: 4, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / f32             |32          |32         |Min: 8, pow2   |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / bf16            |16          |16         |Min: 8, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / bf16            |16          |16         |Min: 16, pow2  |    gfx90a, gfx940+, gfx11  |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / bf16            |32          |32         |Min: 4, pow2   |          gfx908            |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / bf16            |32          |32         |Min: 8, pow2   |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+
|bf16 / bf16 / bf16*           |16          |16         |Min: 8, pow2   |                            |
|                              |            |           |               |                            |
|                              |32          |32         |Min: 4, pow2   |                            |
+------------------------------+------------+-----------+---------------+----------------------------+
|f32 / f32 / f32               |16          |16         |Min: 4, pow2   |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|f32 / f32 / f32               |32          |32         |Min: 2, pow2   |           gfx9             |
+------------------------------+------------+-----------+---------------+----------------------------+
|f64** / f64** / f64**         |16          |16         |Min: 4, pow2   |      gfx90a, gfx940+       |
+------------------------------+------------+-----------+---------------+----------------------------+

*= Matrix unit accumulation is natively 32-bit precision and is converted to the desired type.

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
