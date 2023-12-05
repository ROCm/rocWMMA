# rocWMMA

rocWMMA is a C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware. rocWMMA makes it easier to break down MMA problems
into fragments and distribute block-wise MMA operations in parallel across GPU wavefronts. Our API
consists of a header library, that you can use to compile MMA acceleration directly into GPU kernel
device code. This can benefit from compiler optimization in the generation of kernel assembly, and
doesn't incur additional overhead costs of linking to external runtime libraries or having to launch
separate kernels.

The rocWMMA API is designed from a *wave* perspective, where block-wise API calls are processed by
GPU threads coordinated together as a group, or a *wave*. Importantly, individual threads may only
access a portion of a fragment, and there is no guaranteed layout or locality of this data. Thread access
to fragment data is analogous to the vector register access of individual threads. Fragments and
block-wise operations can therefore be thought of as a wave of grouped data buffers and operations.

Thread blocks that contain multiple waves have the added benefit of resource sharing and
cooperation. For rocWMMA, this is especially useful for moving data that may be shared across
multiple waves. The rocWMMA `Coop` API facilitates cooperation between multiple waves when
loading and storing opaque data fragments. The order and locality of fragment data contribution per
wave is not guaranteed, and can be thought of as a *distributed wave*. Typically, waves cooperate in
moving data from global memory to shared memory (LDS), loading their own full fragment copies
from LDS.

Memory addresses are treated as 1D arrays. The rocWMMA API can opaquely handle moving data
between both global and shared memory address locations. The library code includes utilities for
mapping 2D grid and matrix coordinates to 1D array coordinates, and supports either row-major or
column-major data layouts. Block-wise dimensions (BlockM,N,K) familiar to block-wise general matrix
multiply (GEMM) product algorithms are supported directly through the availability of matrix
instructions for the target architecture. Likewise, you can vary mixed-precision data types for input,
output, and accumulation fragments (as listed in the supported configurations section).

rocWMMA is a header library that includes test and sample projects to validate and demonstrate API
use. GEMM is used as primary validation given the heavy precedent for the library. However, we are
expanding our portfolio to demonstrate more diverse rocWMMA uses.

## Requirements

You can use rocWMMA with the following hardware:

* AMD CDNA class GPU featuring matrix core support: gfx908, gfx90a as 'gfx9'
* AMD RDNA3 class GPU featuring AI acceleration support: gfx1100, gfx1101, gfx1102 as 'gfx11'

```note
Double precision FP64 data type support requires gfx90a
```

rocWMMA software requirements include:

* ROCm stack minimum version 5.4
* rocm-cmake minimum version 0.8.0 for ROCm 5.3
* C++ 17
* CMake >=3.6
* OpenMP

Optional:

* rocBLAS minimum version 2.46.0 for ROCm 5.4 (for rocBLAS validation and benchmarks)
* Doxygen (for building documentation)

## Documentation

To build our documentation locally, use the following commands.

```bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Currently supported configurations

* Wave Size: Wave32 (gfx11), Wave64 (gfx9)
* Matrix Layout <LayoutA, LayoutB, Layout C, LayoutD> (N = col major, T = row major)

```bash
<N, N, N, N>, <N, N, T, T>

<N, T, N, N>, <N, T, T, T>

<T, N, N, N>, <T, N, T, T>

<T, T, N, N>, <T, T, T, T>
```

* Thread Block Sizes <TBlockX, TBlockY>

```note
TBlockX must be a multiple of WaveSize
```

For GEMM, rocWMMA focuses on thread blocks of up to 4 waves for optimal resource usage and
occupancy. Larger thread block sizes are possible but are not officially supported.

```bash
WS = Wave Size

<WS, 1>, <WS, 2>, <WS, 4>

<WS * 2, 1>, <WS * 2, 2>

<WS * 4, 1>
```

* Data Types <Ti / To / Tc> = <Input type / Output Type / Compute Type>

    Input Type = Matrix A/B

    Output Type = Matrix C/D

    Compute Type = math / accumulation type

```note
gfx11 only supports BlockM/N = 16
```

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

## API functions

### `fill_fragment`

Broadcasts a desired value to all elements in the fragment.

### `load_matrix_sync` / `store_matrix_sync`

Loads data from memory according to the matrix layout.

* Matrix A layout: Loads and stores matrix columns in the K direction (Matrix A = M x K,
  fragA = BlockM x BlockK)
* Matrix B layout: Loads and stores matrix rows in the K direction (Matrix B = K x N,
  fragB = BlockK x BlockN)
* Matrix C layout: Loads and stores matrix rows in a vector width of 4 (Matrix C = M x N,
  fragAcc = BlockM x BlockN)

Fragments are stored in packed registers in optimal load and store patterns. In-register elements have
no guaranteed order, which optimizes loading and storing efficiency.

### `mma_sync`

The MMA operation is performed on fragment data. The outer product of Fragment A elements with
Fragment B elements is added to the accumulator fragment.

### `synchronize_workgroup`

Flow control for synchronization across multiple wavefronts in a workgroup. It also ensures the
synchronization of shared and global memory accesses across wavefronts.

### `load_matrix_coop_sync` / `store_matrix_coop_sync`

Loads data from memory according to the matrix layout. Splits operation among wave members of a
workgroup.

* Matrix A: Layout loads and stores matrix columns in the K direction (Matrix A = M x K,
  fragA = BlockM x BlockK)
* Matrix B: Layout loads and stores matrix rows in the K direction (Matrix B = K x N,
  fragB = BlockK x BlockN)
* Matrix C: Layout loads and stores matrix rows in vector width of 4 (Matrix C = M x N,
  fragAcc = BlockM x BlockN)

Each contributing wave handles partial operation data, so split parameters should be consistent across
subsequent operations. For example, cooperatively moving data from global to shared memory should
use the same split parameters for the global load and subsequent local store.

## Contributing to the code

1. Create and track a rocWMMA fork.
2. Clone your fork:

    ```bash
    git clone -b develop https://github.com/<your_fork>/rocWMMA.git .
    .githooks/install
    git checkout -b <new_branch>
    ...
    git add <new_work>
    git commit -m "What was changed"
    git push origin <new_branch>
    ...
    ```

3. Create a pull request to the ROCmSoftwarePlatform/rocWMMA develop branch.
4. Await CI and approval feedback.
5. Once approved, merge.

```note
You must install GitHooks because there are triggers for Clang formatting in commits.
```

## Build with CMake

### Project options

|Option|Description|Default value|
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

By default, the project is configured in release mode and is linked against rocBLAS for validating
results. Here are some configuration examples:

|Configuration|Command|
|---|---|
|Basic|`CC=hipcc CXX=hipcc cmake -B<build_dir> .`|
|Targeting gfx908|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`

```warning
The build time for all projects can take several minutes.
```

### Tips to reduce tests compile time

* Target a specific GPU (e.g., `gfx908:xnack-`)
* Use lots of threads (e.g., `-j64`)
* Select `ROCWMMA_BUILD_ASSEMBLY=OFF`
* Select `ROCWMMA_BUILD_DOCS=OFF`
* Select `ROCWMMA_BUILD_EXTENDED_TESTS=OFF`
* Specify `ROCWMMA_BUILD_VALIDATION_TESTS` or `ROCWMMA_BUILD_BENCHMARK_TESTS` as ON, and the other as OFF
* For investigating particular kernels, build the ad hoc test with the parameters you are interested in
* Manually build specific tests:

    ```bash
    cd <build_dir>
    make <target_name> -j64
    ```

    Where `<target_name>` is one of the following:

    |`<target_name>`|Description|
    |---|---|
    |`rocwmma_unit_tests`|Build all rocWMMA unit tests|
    |`rocwmma_gemm_tests_validate`|Build all GEMM validation tests|
    |`rocwmma_gemm_tests_bench`|Build all GEMM benchmark tests|
    |`rocwmma_dlrm_tests_validate`|Build all deep learning recommendation model (DLRM) validation tests|
    |`rocwmma_dlrm_tests_bench`|Build all DLRM benchmark tests|
    |`rocwmma_samples`|Build all rocWMMA samples|
    |Individual target name (`contamination_test`, `simple_sgemm`, etcetera)|Build individual rocWMMA test or sample|

* Manually reduce tests or test case coverage

## Run unit tests

rocWMMA library features are showcased, validated, and benchmarked if applicable in GoogleTest
applications.

With CTest enabled, you can run the entire test suite by running CTest in the build folder:

```bash
cd <build_dir>
ctest --output-on-failure
```

Otherwise, individual tests can be run as follows:

### Contamination test

Unit tests for loading and storing APIs to verify that data boundaries are not crossed and pristine data
remain untouched.

Run the validation:

```bash
<build_dir>/test/unit/contamination_test
```

### Cross-lane ops test

Unit tests for vector cross-lane operations.

Run the validation:

```bash
<build_dir>/test/unit/cross_lane_ops_test
```

### Fill fragment test

Tests the `rocwmma::fill_fragment` API function for all supported configurations. Tests broadcasting of a
desired value to all elements in the fragment.

Run the validation:

```bash
<build_dir>/test/unit/fill_fragment_test
```

### I/O shape test

Unit test for I/O shape meta-data generation on host machine

Run the validation:

```bash
<build_dir>/test/unit/io_shape_test
```

### I/O traits test

Unit test for I/O traits metadata generation on host machine.

Run the validation:

```bash
<build_dir>/test/unit/io_traits_test
```

### Layout test

Unit tests for the internal collect and scatter matrix element to register mapping transforms.

Run the validation:

```bash
<build_dir>/test/unit/layout_test
```

### Load and store matrix sync test

Tests the `rocwmma::load_matrix_sync` and `rocwmma::store_matrix_sync` API functions for all
supported configurations. Tests proper emplacement of data during loads and stores.

Run the validation:

```bash
<build_dir>/test/unit/load_store_matrix_sync_test
<build_dir>/test/unit/load_store_matrix_coop_sync_test
```

### Map util test

Unit tests for the utility class used to calculate transforms and offsets between grid, matrix, and data
coordinate systems.

Run the validation:

```bash
<build_dir>/test/unit/map_util_test
```

### Vector iterator test

Unit tests for internal vector iteration and navigation during access and storage.

Run the validation:

```bash
<build_dir>/test/unit/vector_iterator_test
```

### Vector test

Unit tests for internal vector storage.

Run the validation:

```bash
<build_dir>/test/unit/vector_test
```

### GEMM tests

Implements a GEMM blocking algorithm using rocWMMA for all supported parametric configurations.
Validates on a CPU algorithm, or rocBLAS if available. Validation runs are performed on a reduced
subset of matrix sizes. Benchmark runs are performed on a larger set of matrix sizes. Extended tests
provide comprehensive parameter coverage and larger matrix sizes.

#### GEMM kernel naming nomenclature

As of rocWMMA v0.8, we've introduced GEMM kernel naming nomenclature to allow compact
representation and quick identification of kernel implementation features. The rocWMMA GEMM test
library includes kernels that support the following features:

```bash
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

* `gemm_PGR0_LB0_MP0_SB_NC`: The simplest blocked GEMM example, which targets one output
  block of matrix multiplication per wave. No prefetch, no lDs usage, default MFMA prioritization, single
  block output and non-collaborative.

* `gemm_PGR0_LB0_MP0_MB_NC`: Implements a multi-block GEMM where each wave is responsible
  for a BlocksX x BlocksY grid of output blocks. No prefetch, no lDs usage, default MFMA prioritization,
  multiple blocks output, and non-collaborative.

* `gemm_PGR1_LB2_MP0_MB_CP_BLK`: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double lDs buffer, default MFMA prioritization, multiple blocks
  output, and is block-tile collaborative in global read and local write.

* `gemm_PGR1_LB2_MP0_MB_CP_WV`: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double lDs buffer, default MFMA prioritization, multiple blocks
  output, and is wave-tile collaborative in global read and local write.

* `gemm_PGR1_LB2_MP0_MB_CP_WG`: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double lDs buffer, default MFMA prioritization, multiple blocks
  output and is macro-tile collaborative in global read and local write.

* `Ad Hoc Test`: An executable that focuses on a specific set of kernel parameters. This is used as a
  quick mock-up of a situational investigation of a particular GEMM kernel.

Validation tests are postfixed with `-validate`. Benchmark tests are postfixed with `-bench`.

Run the validation tests:

```bash
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC-validate
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WV-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WG-validate
```

Run the benchmark only:

Note that benchmark runs can take **several hours** to complete.

```bash
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC-bench
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WV-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_WG-bench
```

Run the ad hoc test:

```bash
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-validate
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-validate
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-validate

<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-bench
<build_dir>/test/gemm/gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-bench
<build_dir>/test/gemm/gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-bench
```

### GEMM test logging arguments

|Compact|Verbose|Description|
|---|---|---|
|-os <output_file>.csv |--output_stream <output_file>.csv| stream GEMM testing output to CSV file |
|  |--omit <int> | omits certain outputs : <code>1 = SKIPPED tests</code> <code>2 - FAILED tests</code> <code>4 - PASSED tests</code> <code>8 - All non-gtest output</code>|

### Tips to reduce run time

* Use GoogleTest filters, target specific test names:

```bash
<test_exe> --gtest_filter=*name_filter*
```

* Manually adjust the test cases coverage
* Use ad hoc tests to focus in on specific parameters
* Select `ROCWMMA_BUILD_EXTENDED_TESTS=OFF`

### Samples

These are standalone, practical use cases for the rocWMMA API. They have minimal dependencies and
represent a targeted application with a fixed set of parameters.

## GEMM

Use MMA to demonstrate rocWMMA API usage in the context of wave-level GEMM computation, in
both simplified and optimized versions.

### Simple GEMM

Simple GEMM algorithm demonstration without LDS memory usage and no transpose.

simple_sgemm calculates D = Alpha * A x B + Beta * C with fp32 inputs and output.

simple_dgemm calculates D = Alpha * A x B + Beta * C with fp64 inputs and output.

simple_hgemm calculates D = Alpha * A x B + Beta * C with fp16 inputs and output.

Includes a simple CPU validation and benchmark.

Run a `simple gemm` sample:

```bash
<build_dir>/samples/simple_sgemm
<build_dir>/samples/simple_dgemm
<build_dir>/samples/simple_hgemm
```

### Peformant SGEMM

To implement and measure performance of Matrix Multiply-Accumulate(D = Alpha * A x B + Beta * C)
with user-defined configurations on a GPU.

It contains the best performant version of the multi-block GEMM algorithm with LDS memory,
macro-tile collaboration, data reuse, and optimized pipeline, configured with the finest parameters for
larger sizes (1K and greater).

perf_sgemm calculates D = Alpha * A x B + Beta * C with fp32 inputs and output.

perf_dgemm calculates D = Alpha * A x B + Beta * C with fp64 inputs and output.

perf_hgemm calculates D = Alpha * A x B + Beta * C with fp16 inputs and output.

Includes a simple CPU validation and benchmark.

Run a `perf gemm` sample:

```bash
<build_dir>/samples/perf_sgemm
<build_dir>/samples/perf_dgemm
<build_dir>/samples/perf_hgemm
```

## GEMV

### SGEMV

Simple MMA with a vector demonstration, without LDS or transpose.

Calculates Y = alpha * (A) * X + beta * Y with fp32 inputs and output.

Includes a simple CPU validation and benchmark.

 A = Matrix of size m * k (col-major)

 X = Vector of size k * 1 (col-major)

 Y = accumulator of size m * 1 (col-major)

Run the `sgemv` sample:

```bash
<build_dir>/samples/simple_sgemv
```

### DGEMV

Simple MMA with a vector demonstration, without LDS or transpose.

Calculates Y = alpha * (A) * X + beta * Y with fp64 inputs and output.

Includes a simple CPU validation and benchmark.

 A = Matrix of size m * k (row-major)

 X = Vector of size k * 1 (col-major)

 Y = accumulator of size m * 1 (row-major)

Run the `dgemv` sample:

```bash
<build_dir>/samples/simple_dgemv
```

## Simple deep learning recommendation model

Simple deep learning recommendation model (DLRM) for machine learning. Implements both forward
and backward passes on fp16 inputs and outputs.

Includes a simple CPU validation and benchmark.

Run the `simple_dlrm` sample:

```bash
<build_dir>/samples/simple_dlrm
```

## hipRTC support

The HIP runtime compilation (hipRTC) environment allows simultaneous compilation, loading, and
running of device code on AMD GPUs. The rocWMMA library is compatible with hipRTC, so you can
leverage it for runtime-generated kernels. A simple GEMM sample is included to demonstrate
compatibility. For more information, refer to the
[HIP API reference](https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/index.html).

```important
The `rocwmma::bfloat16_t` data type is not currently supported in hipRTC.
```

### hipRTC GEMM sample

A simple GEMM algorithm demonstrating runtime compilation (hipRTC) compatibility. Calculates
D = Alpha * A x B + Beta * C with mixed-precision fp16 inputs and fp32 output. Includes code
compilation, module loading, and kernel launch with the hipRTC API, with simple CPU validation and
benchmarking.

Run the `hipRTC_gemm` sample:

```bash
<build_dir>/samples/hipRTC_gemm
```
