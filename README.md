# WMMA

AMD's C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications on GPU using MFMA hardware cores.

## Minimum Requirements
* Rocm stack minimum version 4.0
* C++ 14
* rocblas (only if rocblas validation is used) https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/tag/rocm-4.0.0
* CMake >3.5 (optional)

## Currently supported configurations (ongoing)

- Matrix Layout <LayoutA, LayoutB, Layout C, LayoutD> (N = col major, T = row major)  
    <N, N, N, N>  
    <N, T, N, N>  
    <T, N, N, N>  
    <T, T, N, N>  
    <N, N, T, T>  
    <N, T, T, T>  
    <T, N, T, T>  
    <T, T, T, T>  

- Thread Block Sizes <TBlockX, TBlockY> =  
**Note: TBlockX must be a multiple of 64 **  
    <64, 1>  
    <64, 2>  
    <64, 4>  
    <128, 1>  
    <128, 2>  
    <128, 4>  
    <256, 1>  
    <256, 2>  
    <256, 4>  

- Data Types <Ti / To / Tc> = <Input type / Output Type / Compute Type> 

    * <f16 / f16 / f32> - MFMA Block Size <BlockM, BlockN, BlockK> =  
    <16, 16, 16>  
    <16, 16, 32>  
    <16, 16, 64>  
    <16, 16, 128>  
    <16, 16, 256>  
    <16, 16, 512>  
    <32, 32, 8>  
    <32, 32, 16>  
    <32, 32, 32>  
    <32, 32, 64>  
    <32, 32, 128>  
    
    * <f16 / f32 / f32> - MFMA Block Size <BlockM, BlockN, BlockK> =  
    <16, 16, 16>  
    <16, 16, 32>  
    <16, 16, 64>  
    <16, 16, 128>  
    <16, 16, 256>  
    <16, 16, 512>  
    <32, 32, 8>  
    <32, 32, 16>  
    <32, 32, 32>  
    <32, 32, 64>  
    <32, 32, 128>  

    * <f32 / f32 / f32> - MFMA Block Size <BlockM, BlockN, BlockK> =  
    <16, 16, 4>  
    <16, 16, 8>  
    <16, 16, 16>  
    <16, 16, 32>  
    <16, 16, 64>  
    <16, 16, 128>  
    <16, 16, 256>  
    <16, 16, 512>  
    <32, 32, 2>  
    <32, 32, 4>  
    <32, 32, 8>  
    <32, 32, 16>  
    <32, 32, 32>  
    <32, 32, 64>  
    <32, 32, 128>  



## Functions
### fill_fragment
Broadcast a desired value to all elements in the fragment.

### load_matrix_sync / store_matrix_sync
Loads data from memory according to Matrix Layout.
- Matrix A layout loads / stores matrix columns in the K direction (Matrix A = M x K, fragA = BlockM x BlockK)
- Matrix B layout loads / stores matrix rows in the K direction (Matrix B = K x N, fragB = BlockK x BlockN)
- Matrix C layout loads / stores matrix rows in vector width of 4 (Matrix C = M x N, fragAcc = BlockM x BlockN)

Fragments are stored in packed registers in optimal load / store patterns. In-register elements have no guaranteed order, which have been optimized for loading / storing efficiency.

### mma_sync
If necessary, data from fragments A and B are transferred to LDS cache for proper-ordering before MFMA operations. To minimize LDS footprint when using larger workgroups with multiple wavefronts, waves will cooperate to re-order data. LDS capacity is also re-used between A and B fragments.

Required LDS space is calculated by:
```
LDSBytes = max(BlockM * blockDim.y, BlockN * blockDim.x / 64) * BlockK * sizeof(InputT)
```

Finally, MFMA accumulation is performed with fragment data. Fragment A cols are multiplied with Fragment B rows and added to the accumulator fragment.

## Contributing to the code
Clone the repo to current directory:
```
git clone -b <branch> https://github.com/ROCmSoftwarePlatform/WMMA.git .
.githooks/install
```

Please don't forget to install the githooks as there are triggers for clang formatting in commits.

## Tests
WMMA features are showcased, validated and benchmarked if applicable in test applications.
Build all: **Warning: Build time for all projects can be lengthy**
```
make -j16

#Run
./FillFragmentTest
./LoadStoreMatrixSyncTest
./MmaSyncTest-cpu
./MmaSyncTest-rocBLAS
./MmaSyncTest-bench
```


### FillFragmentTest
Tests the wmma::fill_fragment function for all supported configurations. Tests broadcasting of a desired value to all elements in the fragment.
Build and run validation:
```
make FillFragmentTest -j4
./FillFragmentTest
```

### LoadStoreMatrixSyncTest
Tests the load_matrix_sync and store_matrix_sync functions for all supported configurations. Tests proper emplacement of data during loads and stores.
Build and run validation:
```
make LoadStoreMatrixSyncTest -j4
./LoadStoreMatrixSyncTest
```

### MmaSyncTest
Implements a simple GEMM using wmma for all supported configurations. Validates on CPU algorithm or rocBLAS if available. Also provides benchmarking data on GEMM performance.
Build and run (CPU validation + benchmark): **Warning CPU validation can be very slow especially for large matrices**
```
make MmaSyncTest-cpu -j4
./MmaSyncTest-cpu
```

Build and run (rocBLAS validation + benchmark):
```
export ROCBLAS_DIR=<path_to_rocblas_dir>  
make MmaSyncTest-rocBLAS -j4  
export LD_LIBRARY_PATH=ROCBLAS_DIR/lib:$LD_LIBRARY_PATH
./MmaSyncTest-rocBLAS
```

Build and run (benchmark only):
```
make MmaSyncTest-bench -j4
./MmaSyncTest-bench
```
