************
Introduction
************

rocWMMA is AMD's C++ library for accelerating mixed precision matrix multiply-accumulate operations
leveraging specialized GPU matrix cores on AMD's latest discrete GPUs.

A C++ API is provided to facilitate decomposition of matrix multiply-accumulate problems into
discretized block fragments and to parallelize block-wise operations across multiple GPU wavefronts.

The API is implemented in GPU device code: it empowers user device kernel code with direct use of GPU matrix cores.
Moreover, this code can benefit from inline compiler optimization passes and does not incur additional
overhead of external runtime calls or extra kernel launches.

======== =========
Acronym  Expansion
======== =========
**GEMM**    **GE**\ neral **M**\ atrix **M**\ ultiply
**WMMA**    **W**\ avefront **M**\ ixed precision **M**\ ultiply **A**\ ccumulate
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========

rocWMMA is written in C++ and may be applied directly in device kernel code. Library code is templated
for modularity and uses available meta-data to provide opportunities for compile-time inferences and optimizations.

The rocWMMA API exposes block-wise data load / store and matrix multiply-accumulate functions appropriately sized
for thread-block execution on data fragments. Matrix multiply-accumulate functionality supports mixed precision inputs
and outputs with native fixed-precision accumulation. The rocWMMA Coop API provides wave/warp collaborations
within the thread-blocks for block-wise data load and stores. Supporting code is required for GPU device
management and for kernel invocation. Kernel code samples and tests provided are built and launched via the HIP
ecosystem within ROCm.

Below is a simple example code for calling rocWMMA functions load_matrix_sync, store_matrix_sync, fill_fragment, mma_sync.

.. code-block:: c++

   #include <hip/hip_ext.h>
   #include <hip/hip_fp16.h>
   #include <hip/hip_runtime.h>

   #include <iostream>
   #include <vector>

   #include <rocwmma/rocwmma.hpp>

   using rocwmma::float16_t;
   using rocwmma::float32_t;

   // Matrix data initialization
   template <typename DataT>
   __host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
   {
       auto ld = n;
       for(int i = 0; i < m; ++i)
       {
           for(int j = 0; j < n; ++j)
           {
                // Generated data
                // Alternate sign every 3 elements
                auto value      = (i * n + j) % 13;
                mat[i * ld + j] = (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
           }
       }
   }

   // Supports BlockM/N square sizes of
   // : 16 x 16
   // : 32 x 32
   const int ROCWMMA_M = 16;
   const int ROCWMMA_N = 16;

   // Supports ROCWMMA_K sizes as
   // : multiples of 16.
   const int ROCWMMA_K = 16;

   // AMDGCN default wave size
   const int WAVE_SIZE = rocwmma::AMDGCN_WAVE_SIZE;

   // Thread block
   // : T_BLOCK_X must be multiple of WAVE_SIZE.
   // Note: Each wave will compute one BLOCK_M x BLOCK_N output block
   // Note: Workgroup will compute
   //  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
   // This thread block will compute (4 x 4 output blocks)
   const int T_BLOCK_X = 4 * WAVE_SIZE;
   const int T_BLOCK_Y = 4;

   // The following device kernel is a naive implementation
   // of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
   // output block of the M x N x K GEMM, generalized as:
   // D = alpha * (A x B) + beta * C
   //
   // In this simplified example, we assume:
   // : A is in row-major format     (m x k)
   // : B is in col-major format     (k x n)
   // : C, D are in row-major format (m x n)
   // : Multiplication is NOT in-place, output is written to D matrix
   // : No LDS required
   //
   // Disclaimer: This is a simplified implementation to demonstrate API usage in
   // context of wave-level GEMM computation, and is not optimized.
   //
   // Launchable device kernel function:
   //
   __global__ void gemm_wmma_d(uint32_t         m,     // matrix free dim m
                               uint32_t         n,     // matrix free dim n
                               uint32_t         k,     // matrix fixed dim k
                               float16_t const* a,     // device data ptr for matrix A
                               float16_t const* b,     // device data ptr for matrix B
                               float32_t const* c,     // device data ptr for matrix C
                               float32_t*       d,     // device data ptr for matrix D
                               uint32_t         lda,   // leading dimension for matrix A
                               uint32_t         ldb,   // leading dimension for matrix B
                               uint32_t         ldc,   // leading dimension for matrix C
                               uint32_t         ldd,   // leading dimension for matrix D
                               float32_t        alpha, // uniform scalar
                               float32_t        beta)  // uniform scalar
   {
       // Create frags with meta-data context for block-wise GEMM decomposition
       // @tp0: fragment context = matrix_a, matrix_b or accumulator
       // @tp1: block size M
       // @tp2: block size N
       // @tp3: block size K
       // @tp4: fragment data type
       // @tp5: data layout = row_major, col_major or void (default)
       auto fragA = rocwmma::fragment<rocwmma::matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, rocwmma::row_major>();
       auto fragB = rocwmma::fragment<rocwmma::matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, rocwmma::col_major>();
       auto fragC   = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();
       auto fragAcc = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

       // Initialize accumulator fragment
       rocwmma::fill_fragment(fragAcc, 0.0f);

        // Tile using a 2D grid
        auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
        auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

        // Target C block
        auto cRow = majorWarp * ROCWMMA_M;
        auto cCol = minorWarp * ROCWMMA_N;

       // Bounds check
       if(cRow < m && cCol < n)
       {
            // fragAcc = A x B
            for(int i = 0; i < k; i += ROCWMMA_K)
            {
                // Load the inputs
                rocwmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
                rocwmma::load_matrix_sync(fragB, b + (i + cCol * ldb), ldb);

                // Matrix multiply - accumulate using MFMA units
                rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
            }

            // Fetch C matrix
            rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

            // D = alpha * A x B + beta * C
            for(int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
            }

            // Store to D
            rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);
        }
   }

   // Host side supporting device mgmt and launch code
   __host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float32_t alpha, float32_t beta)
   {
       // Problem size check
       if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
           || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
        {
            std::cout << "Unsupported size!\n";
            return;
        }

        int lda = k;
        int ldb = k;
        int ldc = n;
        int ldd = ldc;

        std::cout << "Initializing host data..." << std::endl;

        // Initialize input matrices
        std::vector<float16_t> matrixA(m * k);
        std::vector<float16_t> matrixB(k * n);
        std::vector<float32_t> matrixC(m * n);
        // Fill outputs with NaN to catch contamination
        std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

        fill(matrixA.data(), m, k);
        fill(matrixB.data(), k, n);
        fill(matrixC.data(), m, n);

        std::cout << "Initializing device data..." << std::endl;

        // Allocate and copy device memory
        float16_t* d_a;
        float16_t* d_b;
        float32_t* d_c;
        float32_t* d_d;

        const size_t bytesA = matrixA.size() * sizeof(float16_t);
        const size_t bytesB = matrixB.size() * sizeof(float16_t);
        const size_t bytesC = matrixC.size() * sizeof(float32_t);
        const size_t bytesD = matrixD.size() * sizeof(float32_t);

        CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
        CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
        CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
        CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

        CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

         auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
         auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
                rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));

         std::cout << "Launching GEMM kernel..." << std::endl;

         hipEvent_t startEvent, stopEvent;
         CHECK_HIP_ERROR(hipEventCreate(&startEvent));
         CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

         hipExtLaunchKernelGGL(gemm_wmma_d,
                          gridDim,
                          blockDim,
                          0, // sharedMemBytes
                          0, // stream
                          startEvent, // Event start
                          stopEvent, // event stop
                          0, // flags
                          m,
                          n,
                          k,
                          d_a,
                          d_b,
                          d_c,
                          d_d,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta);

         auto elapsedTimeMs = 0.0f;
         CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
         CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
         CHECK_HIP_ERROR(hipEventDestroy(startEvent));
         CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

         // Release device memory
         CHECK_HIP_ERROR(hipFree(d_a));
         CHECK_HIP_ERROR(hipFree(d_b));
         CHECK_HIP_ERROR(hipFree(d_c));
         CHECK_HIP_ERROR(hipFree(d_d));

         std::cout << "Finished!" << std::endl;
   }

   int main()
   {
       gemm_test(256, 256, 256, 2.1f, 2.1f);
       return 0;
   }

Synchronous API
^^^^^^^^^^^^^^^

In general, rocWMMA API functions ( load_matrix_sync, store_matrix_sync, mma_sync ) are assumed to be synchronous when
used in context of global memory.

When using these functions in the context of shared memory (e.g. LDS memory), additional explicit workgroup synchronization
may be required due to the nature this memory usage.


Supported Data Types
^^^^^^^^^^^^^^^^^^^^

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>

where

Input Type = Matrix A/B

Output Type = Matrix C/D

Compute Type = math / accumulation type

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

*= matrix unit accumulation is natively 32 bit precision, and is converted to desired type.

**= f64 datatype is only supported on MI-200 class AMDGPU and successors.


Supported Matrix Layouts
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


rocWMMA Datatypes
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



VectorStorage
'''''''''''''

.. doxygenstruct:: rocwmma::detail::VectorStorage


IOConfig
''''''''''''

.. doxygenstruct:: rocwmma::IOConfig


IOShape
''''''''''''

.. doxygenstruct:: rocwmma::IOShape


rocWMMA Enumeration
^^^^^^^^^^^^^^^^^^^

   Enumeration constants have numbering that is consistent with standard C++ libraries.


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
