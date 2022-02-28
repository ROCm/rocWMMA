************
Introduction
************

rocWMMA is a `BLAS's Level 3 GEMM <http://www.netlib.org/blas/#_level_3>`__ implementation on top of AMD's Radeon Open Compute `ROCm <https://rocm.github.io/install.html>`__ runtime and toolchains.
rocWMMA is leveraging AMD's GPU hardware matrix cores through HIP and optimized for AMD's latest discrete GPUs.

======== =========
Acronym  Expansion
======== =========
**GEMM**    **GE**\ neral **M**\ atrix **M**\ ultiply
**WMMA**    **W**\ arp **M**\ atrix **M**\ ultiply **A**\ ccumulate
**ROCm**    **R**\ adeon **O**\ pen E\ **C**\ osyste\ **m**
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========


The aim of rocWMMA is to provide:

- functionality similar to Legacy BLAS's Level 3 GEMM, adapted to run on GPUs
- high performance robust implementation

rocWMMA is written in C++14 and HIP. It uses AMD's ROCm runtime to run on GPU devices.

Specifically, the rocWMMA library enhances the portability of CUDA WMMA code to AMD's heterogeneous platform
and provides an interface to use underlying hardware matrix multiplication (MFMA) units.

The ROCWMMA API exposes memory and MMA (Matrix Multiply Accumulate) functions that operate on blocks, or 'fragments' of data
appropriately sized for warp (thread block) execution.

ROCWMMA code is templated for componentization and for providing ability to make compile-time optimizations
based on available meta-data.

rocWMMA library is an ongoing Work-In-Progress (WIP).

The official rocWMMA API is the C99 API defined in rocwmma.h and therefore the use of any other public symbols is discouraged. All other C/C++ interfaces may not follow a deprecation model and so can change without warning from one release to the next.

rocWMMA uses the provided device memory functions to allocate device memory for the input and output matrices and filling/copying it with appropriate input data.

rocWMMA functions run on the host and they call HIP to launch rocWMMA kernels that run on the device in a HIP stream.

Before calling a rocWMMA function arrays must be copied to the device. Integer scalars like m, n, k are stored on the host. Floating point scalars like alpha and beta can be on host or device.

Below is a simple example code for calling rocwmma functions load_matrix_sync, store_matrix_sync, fill_fragment, mma_sync.

.. code-block:: c++

   #include <hip/hip_ext.h>
   #include <hip/hip_fp16.h>
   #include <hip/hip_runtime.h>

   #include <iostream>
   #include <vector>

   #include "WMMA.h"

   using rocwmma::float16_t;
   using rocwmma::float32_t;
   using rocwmma::float64_t;

   // Matrix data initialization
   template <typename DataT>
   __host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
   {
       auto ld = n;
       for(int i = 0; i < m; ++i)
       {
           for(int j = 0; j < n; ++j)
           {
               // Ascending order for each neighboring element.
                // Alternate sign for even / odd
                auto value      = (i * n + j) % 13;
                mat[i * ld + j] = (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
           }
       }
   }


   // Supports ROCWMMA_M/N square sizes of
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
   const int T_BLOCK_X = 4 * WAVE_SIZE;
   const int T_BLOCK_Y = 4;

   // The following device kernel is a naive implementation
   // of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
   // output block of the M x N x K GEMM, generalized as:
   // D = alpha * (A x B) + beta * C
   //
   // In this simplified example, we assume:
   // : A is in row-major format     (M x K)
   // : B is in col-major format     (K x N)
   // : C, D are in row-major format (M x N)
   // : Multiplication is NOT in-place, output is written to D matrix
   // : No LDS required
   //
   // Note: This is a simplified implementation to demonstrate API usage in
   // context of wave-level GEMM computation, and is not optimized.
   __global__ void gemm_wmma_d(uint32_t         m,
                               uint32_t         n,
                               uint32_t         k,
                               float16_t const* a,
                               float16_t const* b,
                               float32_t const* c,
                               float32_t*       d,
                               uint32_t         lda,
                               uint32_t         ldb,
                               uint32_t         ldc,
                               uint32_t         ldd,
                               float32_t        alpha,
                               float32_t        beta)
   {
       // Create frags
       auto fragA = rocwmma::
           fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float16_t, rocwmma::row_major>();
       auto fragB = rocwmma::
           fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float16_t, rocwmma::col_major>();
       auto fragC   = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float32_t>();
       auto fragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float32_t>();

       rocwmma::fill_fragment(fragAcc, 0.0f);

       // Tile using a 2D grid
       auto warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
       auto warpN = (blockIdx.y * blockDim.y + threadIdx.y);

       // Target C block
       auto cRow = warpM * WMMA_M;
       auto cCol = warpN * WMMA_N;

       // Bounds check
       if(cRow < m && cCol < n)
       {
           // fragAcc = A x B
           for(int i = 0; i < k; i += WMMA_K)
           {
               int aRow = cRow;
               int aCol = i;

               int bRow = i;
               int bCol = cCol;

               // Load the inputs
               rocwmma::load_matrix_sync(fragA, a + (aRow * lda + aCol), lda);
               rocwmma::load_matrix_sync(fragB, b + (bRow + bCol * ldb), ldb);

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

   __host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float32_t alpha, float32_t beta)
   {
       // Bounds check
       if((m < (WMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (WMMA_N * T_BLOCK_Y) || k < WMMA_K)
           || (m % WMMA_M || n % WMMA_N || k % WMMA_K))
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
         auto gridDim  = dim3(rocwmma::ceilDiv(m, WMMA_M * T_BLOCK_X / WAVE_SIZE),
                rocwmma::ceilDiv(n, WMMA_N * T_BLOCK_Y));

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

rocWMMA API functions ( load_matrix_sync, store_matrix_sync, mma_sync ) will be synchronous.


Supported Data Types
^^^^^^^^^^^^^^^^^^^^

All WMMA API function supports the native datatypes : float = f32, double = f64, _Float16 = f16, int8, uint8, int16, int32, uint32 and
non-native datatypes : h16 = __half, bf16 = bfloat16


Supported Matrix Layouts
^^^^^^^^^^^^^^^^^^^^^^^^

 (N = col major, T = row major)

<LayoutA, LayoutB, Layout C, LayoutD>
<N, N, N, N>
<N, N, T, T>
<N, T, N, N>
<N, T, T, T>
<T, N, N, N>
<T, N, T, T>
<T, T, N, N>
<T, T, T, T>


-----------------
Using rocWMMA API
-----------------

This section describes how to use the rocWMMA library API.


rocWMMA Datatypes
^^^^^^^^^^^^^^^^^

matrix_a
''''''''

.. doxygenstruct:: matrix_a


matrix_b
''''''''

.. doxygenstruct:: matrix_b


accumulator
'''''''''''

.. doxygenstruct:: accumulator


row_major
'''''''''

.. doxygenstruct:: row_major


col_major
'''''''''

.. doxygenstruct:: col_major


VecT
''''

.. doxygenclass:: VecT


VectorStorage
'''''''''''''

.. doxygenclass:: VectorStorage


rocWMMA Enumeration
^^^^^^^^^^^^^^^^^^^

   Enumeration constants have numbering that is consistent with standard C libraries.


layout_t
''''''''''''

.. doxygenenum:: layout_t


rocWMMA Helper functions
^^^^^^^^^^^^^^^^^^^^^^^^

Auxiliary Functions
'''''''''''''''''''

.. doxygenfunction:: dataTypeToString
.. doxygenfunction:: epsilon
.. doxygenfunction:: infinity
.. doxygenfunction:: max
.. doxygenfunction:: min
.. doxygenfunction:: quiet_NaN
.. doxygenfunction:: signaling_NaN
.. doxygenfunction:: lowest

Device Memory Allocation Functions
''''''''''''''''''''''''''''''''''

.. doxygenfunction:: hipMalloc
.. doxygenfunction:: hipMemcpy
.. doxygenfunction:: hipFree

Event Synchronization Functions
'''''''''''''''''''''''''''''''

.. doxygenfunction:: hipEventCreate
.. doxygenfunction:: hipEventSynchronize
.. doxygenfunction:: hipEventElapsedTime
.. doxygenfunction:: hipEventDestroy

rocWMMA API functions
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: fill_fragment
   :outline:
.. doxygenfunction:: load_matrix_sync
   :outline:
.. doxygenfunction:: store_matrix_sync
   :outline:
.. doxygenfunction:: mma_sync
   :outline:
.. doxygenfunction:: synchronize_workgroup
   :outline:
.. doxygenfunction:: load_matrix_coop_sync
   :outline:
.. doxygenfunction:: store_matrix_coop_sync
   :outline:
