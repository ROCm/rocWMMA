# rocWMMA

Welcome! rocWMMA is a C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware. rocWMMA makes it easier to break down MMA problems
into fragments and distribute block-wise MMA operations in parallel across GPU wavefronts. The API
consists of a header library, that can be used to compile MMA acceleration directly into GPU kernel
device code. This can benefit from compiler optimization in the generation of kernel assembly, and
doesn't incur additional overhead costs of linking to external runtime libraries or having to launch
separate kernels.

rocWMMA includes sample projects to validate and demonstrate API usage. These include simple GEMMs,
performant GEMMs, DLRM, GEMV and hipRTC integration.

The test suite includes validation and benchmarking projects that focus on unit testing, GEMMs and DLRM.

## Requirements

rocWMMA currently supports the following AMDGPU architectures:

* CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx940, gfx940, gfx942 as 'gfx9'
* RDNA3 class GPU featuring AI acceleration support: gfx1100, gfx1101, gfx1102 as 'gfx11'

Dependencies:

* Minimum ROCm version support is 6.0.
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum rocBLAS version support is rocBLAS 4.0.0 for ROCm 6.0* (or ROCm packages rocblas and rocblas-dev).
* Minimum HIP runtime version support is 4.3.0 (or ROCm package ROCm hip-runtime-amd).
* Minimum LLVM OpenMP runtime dev package version support is 10.0 (available as ROCm package rocm-llvm-dev).

```note::
    * = if using rocBLAS for validation.

    It is best to use available ROCm packages from the same release where applicable.
```

## Build with CMake

For more detailed information, please refer to the [rocWMMA installation guide](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/installation.html).

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
|ROCWMMA_USE_SYSTEM_GOOGLETEST|Use system Google Test library instead of downloading and building it|OFF (requires ROCWMMA_BUILD_TESTS=ON)|

### Example configurations

By default, the project is configured in release mode and is linked against rocBLAS for validating
results. Here are some configuration examples:

|Configuration|Command|
|---|---|
|Basic|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> .`|
|Targeting gfx908|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`

## Documentation

For more comprehensive documentation on installation, samples and test contents, API reference and programmer's guide you can build the documentation locally using the following commands:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

The HTML documentation can be viewed in your browser by opening docs/_build/html/index.html result.

The latest official documentation for rocWMMA is available at:
[https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html).


## Contributing to the rocWMMA Library

Community collaboration is encouraged! If you are considering contributing, please follow the [rocWMMA Contribution Guide](https://github.com/ROCm/rocWMMA/CONTRIBUTING.md) to get started.
