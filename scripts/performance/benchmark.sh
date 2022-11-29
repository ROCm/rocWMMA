#!/usr/bin/env bash
# Copyright (c) 2022 Advanced Micro Devices, Inc.

set -eux

# ensure this script is in the cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

output_dir=rocwmma-benchmarks
build_dir=../../build/test/gemm/

if [ -d "$build_dir" ]; then
  # setup output directory for benchmarks 
  mkdir -p "$output_dir"

  gemm_bench=["gemm_PGR0_LB0_MP0_SB_NC", "gemm_PGR0_LB0_MP0_MB_NC", "gemm_PGR1_LB2_MP0_MB_CP"]
  
  # run benchmarks
  for f in gemm_bench; do
    if [[ -e build_dir/$f && ! -L build_dir/$f ]]; then
      mkdir $output_dir/rocWMMA_$f
      $build_dir$f"-bench" -o "$output_dir/rocWMMA_$f/${f}-benchmark.csv"
    fi
  done
fi
