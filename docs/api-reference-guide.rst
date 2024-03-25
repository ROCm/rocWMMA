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

When using these functions in the context of shared memory (e.g. LDS memory), additional explicit workgroup synchronization
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

+------------------------------+------------+-----------+---------------+
|Ti / To / Tc                  |BlockM      |BlockN     |BlockK         |
+==============================+============+===========+===============+
|i8 / i32 / i32                |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|i8 / i32 / i32                |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|i8 / i8 / i32                 |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|i8 / i32 / i32                |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|f16 / f32 / f32               |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|f16 / f32 / f32               |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|f16 / f16 / f32               |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|f16 / f16 / f32               |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|f16 / f16 / f16*              |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|f16 / f16 / f16*              |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|__half / f32 / f32            |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|__half / f32 / f32            |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|__half / __half / f32         |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|__half / __half / f32         |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|__half / __half / __half*     |16          |16         |Min: 16, pow2  |
+------------------------------+------------+-----------+---------------+
|__half / __half / __half*     |32          |32         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / f32 / f32              |16          |16         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / f32 / f32              |32          |32         |Min: 4, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / bf16 / f32             |16          |16         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / bf16 / f32             |32          |32         |Min: 4, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / bf16 / bf16*           |16          |16         |Min: 8, pow2   |
+------------------------------+------------+-----------+---------------+
|bf16 / bf16 / bf16*           |32          |32         |Min: 4, pow2   |
+------------------------------+------------+-----------+---------------+
|f32 / f32 / f32               |16          |16         |Min: 4, pow2   |
+------------------------------+------------+-----------+---------------+
|f32 / f32 / f32               |32          |32         |Min: 2, pow2   |
+------------------------------+------------+-----------+---------------+
|f64** / f64** / f64**         |16          |16         |Min: 4, pow2   |
+------------------------------+------------+-----------+---------------+

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
