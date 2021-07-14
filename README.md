# WMMA

AMD's C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications on GPU using MFMA hardware cores.

## Minimum Requirements
* Rocm stack minimum version 4.0
* C++ 14
* rocblas (only if rocblas validation is used) https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/tag/rocm-4.0.0
* CMake >=3.5 (optional)

## Currently supported configurations (ongoing)

- Matrix Layout <LayoutA, LayoutB, Layout C, LayoutD> (N = col major, T = row major)

    <N, N, N, N>,  <N, N, T, T>
    
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
MFMA accumulation is performed with fragment data. Fragment A cols are multiplied with Fragment B rows and added to the accumulator fragment.

## Contributing to the code
Clone the repo to current directory:
```
git clone -b <branch> https://github.com/ROCmSoftwarePlatform/WMMA.git .
.githooks/install
```

Please don't forget to install the githooks as there are triggers for clang formatting in commits.


## Build with CMake

By default, the project is configured as Release mode, and is linked against rocBLAS for validating results.
Here are some of the examples for the configuration:
|Configuration|Command|
|---|---|
|Basic|`CC=hipcc CXX=hipcc cmake -B<build_dir> .`|
|Targeting MI100|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DWMMA_VALIDATE_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j`

**Warning: Build time for all projects can take several minutes**

Tips to save compiling time:
- Target a specific GPU (default = MI100, MI200+/-)
- Use lots of threads (e.g. -j32)
- Manually reduce test(s) and/or test cases


## Unit tests with CTest

CTest is a testing tool distributed as a part of CMake. The unit tests can be run by:

```
cd build
ctest
```

## Manually Running Tests
WMMA features are showcased, validated and benchmarked if applicable in test applications.

### FillFragmentTest
Tests the wmma::fill_fragment function for all supported configurations. Tests broadcasting of a desired value to all elements in the fragment.
Run validation:
```
<build_dir>/test/FillFragmentTest
```

### LoadStoreMatrixSyncTest
Tests the load_matrix_sync and store_matrix_sync functions for all supported configurations. Tests proper emplacement of data during loads and stores.
Run validation:
```
<build_dir>/test/LoadStoreMatrixSyncTest
```

### MmaSyncTest
Implements a simple GEMM using wmma for all supported configurations. Validates on CPU algorithm or rocBLAS if available. Validation runs are performed on a reduced subset of matrix sizes. Benchmark only runs are on a larger set of matrix sizes.

Run CPU validation + benchmark: **CPU validation can be very slow especially for large matrices**
```
<build_dir>/test/MmaSyncTest-cpu
```

Run rocBLAS validation + benchmark:
```
./MmaSyncTest-rocBLAS
```

Run benchmark only: **Benchmark runs typically take 3-4 hrs to complete**
```
./MmaSyncTest-bench
```
