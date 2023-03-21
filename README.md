# rocWMMA

AMD's C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA) operations leveraging AMD GPU hardware. rocWMMA provides a C++ API to facilitate breaking down matrix multiply-accumulate problems into fragments and distributes block-wise MMA operations in parallel across GPU wavefronts. The API consists of a header library such that MMA acceleration may be compiled directly into GPU kernel device code. This can benefit from compiler optimization in the generation of kernel assembly, and does not incur additional overhead costs of linking to external runtime libraries or having to launch separate kernels.

Conceptually the rocWMMA API is designed from a 'wave' perspective, such that block-wise API calls are processed by GPU threads coordinated together as a group, or a 'wave'. Importantly, individual threads may only access a portion of a fragment, and there is no guaranteed layout or locality of this data. Thread access to fragment data is analogous to vector register access of individual threads. Fragments and block-wise operations therefore can be thought of in a 'wave' perspective as grouped data buffers and operations.

Thread blocks that contain multiple waves have the added benefit of resource sharing and cooperation. For rocWMMA, this is especially useful for moving data that may be shared across multiple waves. The rocWMMA Coop API facilitates cooperation of multiple waves in loading and storing opaque fragments of data. The order and locality of fragment data contribution per wave is not guaranteed, and can be thought of in a 'distributed wave' perspective. Typically waves cooperate in moving data from global memory to shared memory (LDS) and load their own full fragment copies from LDS.

Memory addresses are treated as 1D arrays and the rocWMMA API can opaquely handle moving data between both global and shared memory address locations. The library code includes utilities for mapping 2D grid / matrix coordinates to 1D array coordinates, and supports either row major or column major data layouts. Block-wise dimensions (BlockM,N,K) familiar to block-wise GEMM matrix product algorithms are supported directly by availability of matrix instructions for the target architecture. Likewise, mixed precision datatypes for input, output and accumulation fragments can be varied as listed below in currently supported configurations.

rocWMMA is released as a header library and also includes test and sample projects to validate and illustrate example usages of the C++ API. GEMM matrix multiplication is used as primary validation given the heavy precedent for the library, however the usage portfolio is growing significantly to demonstrate different ways rocWMMA may be consumed.

## GPU Support
* AMD CDNA class GPU featuring matrix core support: gfx908, gfx90a as 'gfx9'

`Note: Double precision FP64 datatype support requires gfx90a`

OR

* AMD RDNA3 class GPU featuring AI acceleration support: gfx1100, gfx1101, gfx1102 as 'gfx11'

## Minimum Software Requirements
* ROCm stack minimum version 5.4
* ROCm-cmake minimum version 0.8.0 for ROCm 5.3
* C++ 17
* CMake >=3.6
* OpenMP

Optional:
* rocBLAS minimum version 2.46.0 for ROCm 5.4 (for rocBLAS validation / benchmarks)
* doxygen (for building documentation)

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Currently supported configurations (ongoing)
- Wave Size: Wave32 (gfx11), Wave64 (gfx9)

- Matrix Layout <LayoutA, LayoutB, Layout C, LayoutD> (N = col major, T = row major)
```
<N, N, N, N>, <N, N, T, T>

<N, T, N, N>, <N, T, T, T>

<T, N, N, N>, <T, N, T, T>

<T, T, N, N>, <T, T, T, T>

```

- Thread Block Sizes <TBlockX, TBlockY>

`Note: TBlockX must be a multiple of WaveSize`

Currently for GEMM, rocWMMA focuses on thread blocks of up to 4 waves for optimal resource usage / occupancy.
Larger thread block sizes are possible but are not officially supported.

```
WS = Wave Size

<WS, 1>, <WS, 2>, <WS, 4>

<WS * 2, 1>, <WS * 2, 2>

<WS * 4, 1>
```

- Data Types <Ti / To / Tc> = <Input type / Output Type / Compute Type>

    Input Type = Matrix A/B

    Output Type = Matrix C/D

    Compute Type = math / accumulation type

`Note: gfx11 only supports BlockM/N = 16`

<table>
    <thead>
      <tr>
        <th>Ti / To / Tc</th>
        <th>BlockM</th>
        <th>BlockN</th>
        <th>BlockK Range<br /> (powers of 2)</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>i8 / i32 / i32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>i8 / i8 / i32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f16 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>f16 / f16 / f16*</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td rowspan=2>*= CDNA native f32 accumulation downcasted to fp16</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
        </tr>
        <tr>
            <td rowspan=2>__half / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>__half / __half / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>__half / __half / __half*</td>
            <td>16</td>
            <td>16</td>
            <td>[16, 256]</td>
            <td rowspan=2>*= CDNA native f32 accumulation downcasted to __half</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[8, 128]</td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / f32 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[8, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[4, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / bf16 / f32</td>
            <td>16</td>
            <td>16</td>
            <td>[8, 256]</td>
            <td></td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[4, 128]</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>bf16 / bf16 / bf16*</td>
            <td>16</td>
            <td>16</td>
            <td>[8, 256]</td>
            <td rowspan=2>*= CDNA native f32 accumulation downcasted to bf16</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[4, 128]</td>
        </tr>
        <tr>
            <td rowspan=2>f32 / f32 / f32*</td>
            <td>16</td>
            <td>16</td>
            <td>[4, 256]</td>
            <td rowspan=2>*= Supported only on gfx9</td>
        </tr>
        <tr>
            <td>32</td>
            <td>32</td>
            <td>[2, 128]</td>
        </tr>
        <tr>
            <td>f64 / f64 / f64*</td>
            <td>16</td>
            <td>16</td>
            <td>[4, 256]</td>
            <td rowspan=2>*= Supported only on gfx90a +</td>
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
Matrix multiply-accumulate operation is performed on fragment data. The outer product of Fragment A elements with Fragment B elements is added to the accumulator fragment.

### synchronize_workgroup
Flow control for synchronization across multiple wavefronts in a workgroup. It also ensures the synchronization of shared/global memory accesses across wavefronts.

## Cooperative API Functions

### load_matrix_coop_sync / store_matrix_coop_sync
Loads data from memory according to Matrix Layout. Splits operation amongst wave members of a workgroup.
- Matrix A layout loads / stores matrix columns in the K direction (Matrix A = M x K, fragA = BlockM x BlockK)
- Matrix B layout loads / stores matrix rows in the K direction (Matrix B = K x N, fragB = BlockK x BlockN)
- Matrix C layout loads / stores matrix rows in vector width of 4 (Matrix C = M x N, fragAcc = BlockM x BlockN)

Each contributing wave handles partial operation data, therefore split parameters should be consistent across subsequent operations. E.g. Moving data from global to shared memory cooperatively should use the same split parameters for the global load and subsequent local store.

## Contributing to the code
1. Create and track a rocWMMA fork.
2. Clone your fork:
```
git clone -b develop https://github.com/<your_fork>/rocWMMA.git .
.githooks/install
git checkout -b <new_branch>
...
git add <new_work>
git commit -m "What was changed"
git push origin <new_branch>
...
```
3. Create a pull request to ROCmSoftwarePlatform/rocWMMA develop branch.
4. Await CI and approval feedback.
5. Once approved, merge!

`Note: Please don't forget to install the githooks as there are triggers for clang formatting in commits.`


## Build with CMake

### Project options
|Option|Description|Default Value|
|---|---|---|
|AMDGPU_TARGETS|Build code for specific GPU target(s)|gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1100;gfx1101;gfx1102|
|ROCWMMA_BUILD_TESTS|Build Tests|ON|
|ROCWMMA_BUILD_SAMPLES|Build Samples|ON|
|ROCWMMA_BUILD_DOCS|Build doxygen documentation from code|OFF|
|ROCWMMA_BUILD_ASSEMBLY|Generate assembly files|OFF|
|ROCWMMA_BUILD_VALIDATION_TESTS|Build validation tests |ON (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_BUILD_BENCHMARK_TESTS|Build benchmark tests |OFF (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_BUILD_EXTENDED_TESTS|Build extended testing coverage |OFF (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_VALIDATE_WITH_ROCBLAS|Use rocBLAS for validation tests|ON (requires ROCWMMA_BUILD_VALIDATION_TESTS=ON)|
|ROCWMMA_BENCHMARK_WITH_ROCBLAS|Include rocBLAS benchmarking data|OFF (requires ROCWMMA_BUILD_BENCHMARK_TESTS=ON)|

### Example configurations
By default, the project is configured as Release mode, and is linked against rocBLAS for validating results.
Here are some of the examples for the configuration:
|Configuration|Command|
|---|---|
|Basic|`CC=hipcc CXX=hipcc cmake -B<build_dir> .`|
|Targeting gfx908|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`

**Warning: Build time for all projects can take several minutes**

### Tips to reduce tests compile time:
- Target a specific GPU (e.g. gfx908:xnack-)
- Use lots of threads (e.g. -j64)
- Select ROCWMMA_BUILD_ASSEMBLY=OFF
- Select ROCWMMA_BUILD_DOCS=OFF
- Select ROCWMMA_BUILD_EXTENDED_TESTS=OFF
- Select either ROCWMMA_BUILD_VALIDATION_TESTS or ROCWMMA_BUILD_BENCHMARK_TESTS as ON, and the other as OFF.
- For investigating particular kernels, build the ad-hoc test with the parameters you are interested in.
- Manually build specific tests:
```
cd <build_dir>
make <target_name> -j64
```
Where `<target_name>` is one of the following:

|`<target_name>`|Description|
|---|---|
|rocwmma_unit_tests|Build all rocWMMA unit tests|
|rocwmma_gemm_tests_validate|Build all GEMM validation tests|
|rocwmma_gemm_tests_bench|Build all GEMM benchmark tests|
|rocwmma_dlrm_tests_validate|Build all DLRM validation tests|
|rocwmma_dlrm_tests_bench|Build all DLRM benchmark tests|
|rocwmma_samples|Build all rocWMMA samples|
|Individual target name (contamination_test, simple_sgemm, etc...)|Build individual rocWMMA test/sample|

- Manually reduce test(s) and/or test case coverage

## Running Unit Tests
rocWMMA library features are showcased, validated and benchmarked if applicable in GTest applications.

With ctest enabled, you may run the entire test suite by running ctest in the build folder (e.g.):
```
cd <build_dir>
ctest --output-on-failure
```
Otherwise, individual tests can be run as below.

### Contamination Test
Unit tests for loading and storing API to verify data boundaries are not crossed and pristine data remains untouched.
Run validation:
```
<build_dir>/test/unit/contamination_test
```

### Cross-Lane Ops Test
Unit tests for vector cross-lane operations.
Run validation:
```
<build_dir>/test/unit/cross_lane_ops_test
```

### Fill Fragment Test
Tests the rocwmma::fill_fragment API function for all supported configurations. Tests broadcasting of a desired value to all elements in the fragment.
Run validation:
```
<build_dir>/test/unit/fill_fragment_test
```

### I/O Shape Test
Unit test for I/O shape meta-data generation on host machine
Run validation:
```
<build_dir>/test/unit/io_shape_test
```

### I/O Traits Test
Unit test for I/O traits meta-data generation on host machine
Run validation:
```
<build_dir>/test/unit/io_traits_test
```

### Layout Test
Unit tests for the internal collect / scatter matrix element to register mapping transforms.
Run validation:
```
<build_dir>/test/unit/layout_test
```

### Load / Store Matrix Sync Test
Tests the rocwmma::load_matrix_sync and rocwmma::store_matrix_sync API functions for all supported configurations. Tests proper emplacement of data during loads and stores.
Run validation:
```
<build_dir>/test/unit/load_store_matrix_sync_test
<build_dir>/test/unit/load_store_matrix_coop_sync_test
```

### Map Util Test
Unit tests for the utility class used to calculate transforms and offsets between grid, matrix and data coordinate systems.
Run validation:
```
<build_dir>/test/unit/map_util_test
```

### Vector Iterator Test
Unit tests for internal vector iteration and navigation during access and storage.
Run validation:
```
<build_dir>/test/unit/vector_iterator_test
```

### Vector Test
Unit tests for internal vector storage.
Run validation:
```
<build_dir>/test/unit/vector_test
```

### GEMM tests
Implements a GEMM blocking algorithm using rocWMMA for all supported parametric configurations. Validates on CPU algorithm or rocBLAS if available. Validation runs are performed on a reduced subset of matrix sizes. Benchmark runs are on a larger set of matrix sizes. Extended tests provide comprehensive parameter coverage and larger matrix sizes.

#### GEMM kernel naming nomenclature
As of rocWMMA v0.8, GEMM kernel naming nomenclature has been introduced to allow compact representation of and quick identifical of kernel implementation features. rocWMMA GEMM test library includes kernels that support the following features:
```
PGR# - Prefetch Global Read lookup stages. PGR0 = no global read prefetch. PGR1 = 1 stage global read prefetch.
LB# - Lds buffer count. LB0 = no lds usage, LB2 = 2 Lds buffers used for swap.
MP# - MFMA instruction priority. MP0 = default MFMA instruction priority of 0. MP1 = raise MFMA instruction priority to 1.
MB - Multiple output blocks targeted per wave
SB - Single output block target per wave
NC - Non-Cooperative load / store
CP - Cooperative load / store
BLK - Cooperative load / store per block tile
WV - Cooperative load / store per wave tile
WG - Cooperative load / store per macro tile
```

**gemm_PGR0_LB0_MP0_SB_NC** is the simplest blocked GEMM example which targets one output block of matrix multiplication per wave. No prefetch, no lds usage, default MFMA prioritization, single block output and non-collaborative.

**gemm_PGR0_LB0_MP0_MB_NC** implements a multi-block GEMM where each wave is responsible for a BlocksX x BlocksY grid of output blocks. No prefetch, no lds usage, default MFMA prioritization, multiple blocks output and non-collaborative.

**gemm_PGR1_LB2_MP0_MB_CP_BLK** implements a multi-block GEMM where each wave is responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to implement a data prefetching pipeline and collaborates with other waves to improve performance. Implements single stage prefetch, double lds buffer, default MFMA prioritization, multiple blocks output and is block-tile collaborative in global read / local write.

**gemm_PGR1_LB2_MP0_MB_CP_WV** implements a multi-block GEMM where each wave is responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to implement a data prefetching pipeline and collaborates with other waves to improve performance. Implements single stage prefetch, double lds buffer, default MFMA prioritization, multiple blocks output and is wave-tile collaborative in global read / local write.

**gemm_PGR1_LB2_MP0_MB_CP_WG** implements a multi-block GEMM where each wave is responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to implement a data prefetching pipeline and collaborates with other waves to improve performance. Implements single stage prefetch, double lds buffer, default MFMA prioritization, multiple blocks output and is macro-tile collaborative in global read / local write.

**Ad Hoc Test** is an executable that focuses on a specific set of kernel parameters. This is used as a quick mock-up of situational investigation of a particular GEMM kernel.

Validation tests are post-fixed with "-validate"

Benchmark tests are post-fixed with "-bench"

Run validation tests:
```
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC-validate
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WV-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WG-validate
```

Run benchmark only: **Benchmark runs can take several hours to complete**
```
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC-bench
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WV-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WG-bench
```

Run ad-hoc test:
```
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-validate
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-validate

<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-bench
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-bench
```

### GEMM test logging arguments:
|Compact|Verbose|Description|
|---|---|---|
|-os <output_file>.csv |--output_stream <output_file>.csv| stream GEMM testing output to CSV file |
|  |--omit <int> | omits certain outputs : <code>1 = SKIPPED tests</code> <code>2 - FAILED tests</code> <code>4 - PASSED tests</code> <code>8 - All non-gtest output</code>|

### Tips to reduce run time:
- Use gtest filters, target specific test names:
```
<test_exe> --gtest_filter=*name_filter*
```
- Manually adjust the test cases coverage.
- Use ad-hoc tests to focus in on specific parameters.
- Select ROCWMMA_BUILD_EXTENDED_TESTS=OFF

### Samples
These are stand-alone real-world use-cases of the rocWMMA API. They have minimal dependencies and represent a targeted application with a fixed set of parameters.

## GEMM
Matrix Multiply Accumulate to demonstrate rocWMMA API usage in context of wave-level GEMM computation, in both simplified and optimized versions.

**Simple GEMM**

Simple GEMM algorithm demonstration without LDS memory usage and no transpose. 

simple_sgemm calculates D = Alpha * A x B + Beta * C with fp32 inputs and output.

simple_dgemm calculates D = Alpha * A x B + Beta * C with fp64 inputs and output.

simple_hgemm calculates D = Alpha * A x B + Beta * C with fp16 inputs and output.

Includes simple CPU validation and benchmark.

Run simple gemm sample:
```
<build_dir>/samples/simple_sgemm
<build_dir>/samples/simple_dgemm
<build_dir>/samples/simple_hgemm

```

**Peformant SGEMM**
To implement and measure performance of Matrix Multiply-Accumulate(D = Alpha * A x B + Beta * C) with user-defined configurations on GPU.
It contains the best performant version of multi-block GEMM algorithm with LDS memory, Macro Tile Collaboration, Data re-use and
Optimized pipeline, configured with the finest parameters for larger sizes (1K and above).

perf_sgemm calculates D = Alpha * A x B + Beta * C with fp32 inputs and output.

perf_dgemm calculates D = Alpha * A x B + Beta * C with fp64 inputs and output.

perf_hgemm calculates D = Alpha * A x B + Beta * C with fp16 inputs and output.

Includes simple CPU validation and benchmark.

Run perf gemm sample:
```
<build_dir>/samples/perf_sgemm
<build_dir>/samples/perf_dgemm
<build_dir>/samples/perf_hgemm
```

## GEMV

**SGEMV**

Simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose. Calculates Y = alpha * (A) * X + beta * Y with mixed precision fp16 inputs and fp32 output. Includes simple CPU validation and benchmark.

 A = Matrix of size m * k (row-major)

 X = Vector of size k * 1 (col-major)

 Y = accumulator of size m * 1 (row-major)

Run sgemv sample:
```
<build_dir>/samples/simple_sgemv
```

**DGEMV**

Simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose. Calculates Y = alpha * (A) * X + beta * Y with fp64 inputs and output. Includes simple CPU validation and benchmark.

 A = Matrix of size m * k (row-major)

 X = Vector of size k * 1 (col-major)

 Y = accumulator of size m * 1 (row-major)

Run dgemv sample:
```
<build_dir>/samples/simple_dgemv
```

## Simple DLRM

Simple Deep Learning Recommendation Model (DLRM) for machine learning. Implements both forward and backward passes on fp16 inputs and outputs. Includes simple CPU validation and benchmark.

Run simple_dlrm sample:
```
<build_dir>/samples/simple_dlrm
```

## hipRTC Support

HIP's run-time compilation (hipRTC) environment allows on-the-fly compilation, loading and execution of device code on AMD GPUs. The rocWMMA library is compatible with hipRTC where it may be leveraged for run-time generated kernels*. A simple GEMM sample has been included to demonstrate compatibility. hipRTC API documentation may be found in the HIP Programming Guide for the latest release located here: https://docs.amd.com/

`*= rocwmma::bfloat16_t datatype not currently supported in hipRTC.`

**hipRTC GEMM sample**

Simple GEMM algorithm demonstrating runtime compilation (hipRTC) compatibility. Calculates D = Alpha * A x B + Beta * C with mixed precision fp16 inputs and fp32 output. Includes code compilation, module loading and kernel launch with hipRTC API, with simple CPU validation and benchmarking.

Run hipRTC_gemm sample:
```
<build_dir>/samples/hipRTC_gemm
```
