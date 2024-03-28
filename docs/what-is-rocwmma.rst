.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _what-is-rocwmma:

*****************
What is rocWMMA?
*****************

rocWMMA where WMMA stands for Wavefront Mixed precision Multiply Accumulate, is AMD's C++ library for accelerating mixed precision matrix multiply-accumulate operations
leveraging specialized GPU matrix cores on AMD's latest discrete GPUs.

The C++ APIs facilitate the decomposition of matrix multiply-accumulate problems into
discretized block fragments and parallelize the block-wise operations across multiple GPU wavefronts.

The API is implemented in the GPU device code, which empowers user device kernel code with direct use of GPU matrix cores.
Moreover, this code can benefit from inline compiler optimization passes and prevent additional
overhead of external runtime calls or extra kernel launches.

As rocWMMA is written in C++, it can be applied directly in the device kernel code. The library code is templated for modularity and uses the available meta-data to provide opportunities for compile-time inferences and optimizations.

The rocWMMA API exposes block-wise data load and store and matrix multiply-accumulate functions appropriately sized for thread-block execution on data fragments. Matrix multiply-accumulate functionality supports mixed precision inputs and outputs with native fixed-precision accumulation. The rocWMMA Coop API provides wave and warp collaborations within the thread blocks for block-wise data load and stores. 
Supporting code is required for GPU device management and kernel invocation. The provided <kernel code samples and tests> are built and launched via the Heterogeneous-Compute Interface for Portability (HIP) ecosystem within ROCm.