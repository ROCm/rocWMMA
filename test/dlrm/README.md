## DLRM WMMA Benchmark

This benchmark evaluates correctness and performance of WMMA using a "dot" operator in the DLRM benchmark

- - - -

#### Build the benchmark

```bash
# Build HIP
HIP_WMMA_DIR=<WMMA dir> make

# Build CUDA
make cuda
```

#### Run the benchmark

```bash
usage: ./dot_based_interact <FP: 16 | 32>
```

Note that FP32 is not supported in the CUDA benchmark.
