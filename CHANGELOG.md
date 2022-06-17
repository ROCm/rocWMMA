# Change Log for rocWMMA
Full documentation for rocWMMA is available at [rocwmma.readthedocs.io](https://rocwmma.readthedocs.io/en/latest/).
## rocWMMA 0.8 for ROCm 5.3.0
## Added
- Added runtime checks to disable tests on non-target GPUS
- Added workgroup aware gemm kernels
- Added workgroup aware validation and benchmark test suite
- Added warmup run to existing tests

### Changed
- Refactored lds_mapping_util into gemm global, local mapping, gemm driver, gemm config and scheduling classes
- Modified resource allocation and tracking of gemm and dlrm buffers
- Improved low-level data loading patterns
- Reduced branching on cooperative load and store
- Updated gemv sample
- Updated gemm sample


## rocWMMA 0.7 for ROCm 5.2.0
### Added
- Added unit tests for DLRM kernels
- Added GEMM sample
- Added DLRM sample
- Added SGEMV sample
- Added unit tests for cooperative wmma load and stores
- Added unit tests for IOBarrier.h
- Added wmma load/ store  tests for different matrix types (A, B and Accumulator)
- Added more block sizes 1, 2, 4, 8 to test MmaSyncMultiTest
- Added block sizes 4, 8 to test MmaSynMultiLdsTest
- Added support for wmma load / store layouts with block dimension greater than 64
- Added IOShape structure to define the attributes of mapping and layouts for all wmma matrix types
- Added CI testing for rocWMMA

### Changed
- Renamed wmma to rocwmma in cmake, header files and documentation
- Renamed library files
- Modified Layout.h to use different matrix offset calculations (base offset, incremental offset and cumulative offset)
- Opaque load/store continue to use incrementatl offsets as they fill the entire block
- Cooperative load/store use cumulative offsets as they fill only small portions for the entire block
- Increased Max split counts to 64 for cooperative load/store
- Moved all the wmma definitions, API headers to rocwmma namespace
- Modified wmma fill unit tests to validate all matrix types (A, B, Accumulator)


## (Unreleased) rocWMMA 0.6
### Added
- Added unit tests for MappingUtil.h
- Added unit tests for Layout.h
- Added unit tests for non-native vector class in Types.h
- Added unit tests for wmma load and store contatmination check
- Added doxygen support for rocWMMA documentation
- Added mfma barrier in IOBarrier.h
- Added a cmake flag to support assembly code generation of wmma kernels
- Added Mma sync test wmma operation with LDS usage
- Added a script to generate the plots of different wmma benchmarks
- Added multi-block kernels with LDS usage
- Added unit tests for multi-block wmma kernels

### Changed
- Modified GLlops calculation to accommodate multiple devices
- Removed half types packing quirk with col major output
- Moved HIP resource management to HipResource class
- Fixed NaN errors during output comparison

## (Unreleased) rocWMMA 0.5
### Added
- Added templatization for the class amdgcn_convert
- Added wmma load, store and fill support for integral datatypes and float64
- Added mfma support for i8
- Added support for bf16_1k mfma instructions
- Added code to identify the card type and its support during runtime

### Changed
- Refactored and simplified IOBroadcast.h
- Modified the fragment interface compatible with NVIDIA's definition
- Modified cmake to create a lean build of rocWMMA library

## (Unreleased) rocWMMA 0.4
### Added
- Fixed compiler issues with new versions of ROCm
- Added CMake support for the library and unit tests
- Integrated unit test with GTest and OpenMP
- Added host overload operators for hfloat16_t
-
### Changed
- Fixed relative error calculation for non-integral data comparison
- Fixed assembly generation of cooperative load and store code
- Sped up compilation time by moving thread block sizes to function arguments rather than template parameters
- Moved all the existing unit tests to test folder
- Moved all the header files to library/include
- Modified Layout.h to use RowNT/ColNT to eliminate LDS usage in mma_sync
- Deprecated buffer load/store and local load/store

## (Unreleased) rocWMMA 0.3
### Added
- Added support for bfloat16 compute type

### Changed
- Renamed __half to hfloat_16 for consistency
- Modified Convert.h to support native to bfloat16 conversion and vice versa
- Modified IOBroadCast.h to incorporate bfloat16 data packing
- Modified IOTraits.h to add bfloat16 packing traits
- Modified MFMA.h to add mfma invocation calls to bfloat16 data
- Modified WMMA Types to include bfloat16_t
- Modified the wmma load, store and mma unit tests to validate bfloat16

## (Unreleased) rocWMMA 0.2
### Added
- Added support for fp16 compute type
- Fixed data comparison operators for fp16 datatypes
- Added direct MFMA support for non-native __half datatype
- Adjusted the vector storage to accommodate the non-native types

### Changed
- Modified Convert.h to support native to __half conversion and vice versa
- Modified IOBroadCast.h to incorporate __half data packing
- Modified IOTraits.h to add __half packing traits
- Modified MFMA.h to add mfma invocation calls to __half data
- Modified WMMA Types to include __half _t
- Modified the wmma load, store and mma unit tests to validate __half

## (Unreleased) rocWMMA 0.1
### Added
- Defined a wmma namespace with the matrix types, memory and layouts supported
- Defined a fragment datatype to control the data transfer between HIP and MFMA.
- Implemented the rocWMMA functions : load_matrix_sync, load_matrix_coop_sync, store_matrix_sync, fill_fragment and mma_sync
- Implemented Types.h to define the supported datatypes.
- Implemented the class IOTraits to define the packing traits for the defined types as WMMA works on the packed registers.
- Added Buffer load, store to support LLVM data instructions
- Added Opaque load, store
- Added Cooperative load, store to optimize the memory overhead
- Added local load, store to perform register packing
- Implemented Convert.h to perform non-native datatype conversion to native types and vice versa
- Added IOBroadcast class to perform the packing of entire input data(multiple registers)
- Implemented IOConfig to set the optimal Input/Output configurations for the rocWMMA matrix types
- Implemented IOPack and IOUnpack to convert the unpacked device memory to packed registers and vice versa
- Added Layout class to define the data layout in matrix space
- Added MFMA to call the low-level MFMA hardware instructions
- Implemented MappingUtil class to map from workgroup configurations to functional wave units
- Add Performance.h to compute GFlops based on hardware configurations
- Add Reference.h to implement the CPU GEMM operation
- Add Utils.h to implement matrix data operations
- Add rocBLASReference.h to invoke the rocBLAS's GEMM function
- Add unit tests to validate the WMMA APIs - load, store, fill and mma
- Add Makefile support to build library and tests
