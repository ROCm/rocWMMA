# WMMA

AMD's C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications on GPU using MFMA hardware cores.

## Minimum Requirements
* Rocm stack minimum version 4.3
* C++ 14
* CMake >=3.5
* OpenMP

Optional:
* rocblas >= 4.0 (if rocBLAS) https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/tag/rocm-4.0.0
* doxygen (if documentation is built)

## Currently supported configurations (ongoing)

- Matrix Layout <LayoutA, LayoutB, Layout C, LayoutD> (N = col major, T = row major)

    <N, N, N, N>, <N, N, T, T>

    <N, T, N, N>, <N, T, T, T>

    <T, N, N, N>, <T, N, T, T>

    <T, T, N, N>, <T, T, T, T>

- Thread Block Sizes <TBlockX, TBlockY> =
**Note: TBlockX must be a multiple of 64 **

    <64, 1>, <64, 2>, <64, 4>

    <128, 1>, <128, 2>, <128, 4>

    <256, 1>, <256, 2>, <256, 4>

- Data Types <Ti / To / Tc> = <Input type / Output Type / Compute Type>

    Input Type = Matrix A/B

    Output Type = Matrix C/D

    Compute Type = math / accumulation type

<table>
    <thead>
      <tr>
        <th>Ti / To / Tc</th>
        <th>BlockM</th>
        <th>BlockN</th>
        <th>BlockK</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>i8 / i32 / i32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>i8 / i8 / i32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f16 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f16 / f16*</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td rowspan=2>*= FMA is natively f32, downcasted to fp16</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
        </tr>
        <tr>
            <td rowspan=2>__half / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>__half / __half / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>__half / __half / __half*</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 16, pow2</td>
            <td rowspan=2>*= FMA is natively f32, downcasted to __half</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 8, pow2</td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 4, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / bf16 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 8, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 4, pow2</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / bf16 / bf16*</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 8, pow2</td>
            <td rowspan=2>*= FMA is natively f32, downcasted to bf16</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 4, pow2</td>
        </tr>
        <tr>
            <td rowspan=2>f32 / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 4, pow2</td>
            <td rowspan=2></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>Min: 2, pow2</td>
        </tr>
        <tr>
            <td>f64 / f64 / f64</td>
            <td>16</td>
            <td>16</td>
            <td>Min: 4, pow2</td>
            <td></td>
        </tr>
    </tbody>
  </table>


## API Functions
### fill_fragment
Broadcast a desired value to all elements in the fragment.

### load_matrix_sync / store_matrix_sync
Loads data from memory according to Matrix Layout.
- Matrix A layout loads / stores matrix columns in the K direction (Matrix A = M x K, fragA = BlockM x BlockK)
- Matrix B layout loads / stores matrix rows in the K direction (Matrix B = K x N, fragB = BlockK x BlockN)
- Matrix C layout loads / stores matrix rows in vector width of 4 (Matrix C = M x N, fragAcc = BlockM x BlockN)

Fragments are stored in packed registers in optimal load / store patterns. In-register elements have no guaranteed order, which have been optimized for loading / storing efficiency.

### mma_sync
MFMA accumulation is performed with fragment data. Fragment A elements are multiplied with Fragment B elements and added to the accumulator fragment.

### synchronize_workgroup
Performs synchronization across multiple wavefronts in a workgroup. It also ensures the synchronization of shared/global memory accesses across wavefronts.

## Contributing to the code
Clone the repo to current directory:
```
git clone -b <branch> https://github.com/ROCmSoftwarePlatform/WMMA.git .
.githooks/install
```

**Please don't forget to install the githooks** as there are triggers for clang formatting in commits.


## Build with CMake

### Project options
|Option|Description|Default Value|
|---|---|---|
|AMDGPU_TARGETS|Build code for specific GPU target(s)|gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+|
|WMMA_BUILD_TESTS|Build Tests|ON|
|WMMA_BUILD_DOCS|Build doxygen documentation from code|OFF|
|WMMA_BUILD_ASSEMBLY|Generate assembly files|OFF|
|WMMA_BUILD_VALIDATION_TESTS|Build validation tests |ON (requires WMMA_BUILD_TESTS=ON)|
|WMMA_BUILD_BENCHMARK_TESTS|Build benchmark tests |ON (requires WMMA_BUILD_TESTS=ON)|
|WMMA_BUILD_EXTENDED_TESTS|Build extended testing coverage |OFF (requires WMMA_BUILD_TESTS=ON)|
|WMMA_VALIDATE_WITH_ROCBLAS|Use rocBLAS for validation tests| ON (requires WMMA_BUILD_VALIDATION_TESTS=ON)|
|WMMA_BENCHMARK_WITH_ROCBLAS|Include rocBLAS benchmarking data| OFF (requires WMMA_BUILD_BENCHMARK_TESTS=ON)|

### Example configurations
By default, the project is configured as Release mode, and is linked against rocBLAS for validating results.
Here are some of the examples for the configuration:
|Configuration|Command|
|---|---|
|Basic|`CC=hipcc CXX=hipcc cmake -B<build_dir> .`|
|Targeting MI100|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DWMMA_VALIDATE_WITH_ROCBLAS=OFF -DWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j`

**Warning: Build time for all projects can take several minutes**

### Tips to reduce compile time:
- Target a specific GPU (default = MI100, MI200+/-)
- Use lots of threads (e.g. -j64)
- Select WMMA_BUILD_ASSEMBLY=OFF
- Select WMMA_BUILD_EXTENDED_TESTS=OFF
- Manually build specific tests:
```
cd <build_dir>
make <target_name> -j64
```
- Manually reduce test(s) and/or test case coverage

## Running Unit Tests
WMMA library features are showcased, validated and benchmarked if applicable in GTest applications.

### FillFragmentTest
Tests the wmma::fill_fragment API function for all supported configurations. Tests broadcasting of a desired value to all elements in the fragment.
Run validation:
```
<build_dir>/test/FillFragmentTest
```

### LoadStoreMatrixSyncTest
Tests the load_matrix_sync and store_matrix_sync API functions for all supported configurations. Tests proper emplacement of data during loads and stores.
Run validation:
```
<build_dir>/test/LoadStoreMatrixSyncTest
```

### ContaminationTest
Unit tests for loading and storing API to verify data boundaries are not crossed and pristine data is not accessed.
Run validation:
```
<build_dir>/test/ContaminationTest
```

### LayoutTest
Unit tests for the internal data layout offset calculations for all supported configurations.
Run validation:
```
<build_dir>/test/LayoutTest
```

### MappingUtilTest
Unit tests for the MappingUtil class used to help with coordinate transforms and offsets calculations in other tests.
Run validation:
```
<build_dir>/test/MappingUtilTest
```

### VectorIteratorTest
Unit tests for internal vector assignments
Run validation:
```
<build_dir>/test/VectorIteratorTest
```

### GEMM tests
Implements a GEMM blocking algorithm using wmma for all supported configurations. Validates on CPU algorithm or rocBLAS if available. Validation runs are performed on a reduced subset of matrix sizes. Benchmark runs are on a larger set of matrix sizes. There are currently 3 variants of GEMM kernels. **MmaSyncTest** is the simplest GEMM example which
targets one output block of matrix multiplication per wave. **MmaSyncTestLds** implements MmaSyncTest with LDS shared
memory to implement data prefetching and movement pipelining to improve performance. **MmaSyncTestCoopLds** is a
slightly more complicated GEMM that reduces LDS footprint by sharing LDS data with other waves in the workgroup.

Run validation tests:
```
<build_dir>/test/gemm/MmaSyncTest-validate
<build_dir>/test/gemm/MmaSyncTestLds-validate
<build_dir>/test/gemm/MmaSyncTestCoopLds-validate
<build_dir>/test/gemm/MmaSyncMultiTest-validate
<build_dir>/test/gemm/MmasyncMultiLdsTest-validate
```

Run benchmark only: **Benchmark runs can take several hours to complete**
```
<build_dir>/test/gemm/MmaSyncTest-bench
<build_dir>/test/gemm/MmaSyncTestLds-bench
<build_dir>/test/gemm/MmaSyncTestCoopLds-bench
<build_dir>/test/gemm/MmaSyncMultiTest-bench
<build_dir>/test/gemm/MmasyncMultiLdsTest-bench
```

### Tips to reduce run time:
- Use gtest filters, target specific test names:
```
<test_exe> --gtest_filter=*name_filter*
```
- Manually adjust the test cases coverage
