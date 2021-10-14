#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.hpp>

#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>

using namespace nvcuda;

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

struct __align__(8) half4 {
  half2 vals[2];
};

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint M_BLOCKS,
          uint K_BLOCKS,
          uint SMEM_STRIDE,
          uint SMEM_STRIDE_ACC,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFP16Kernel(const __half *__restrict input,
                                                                               __half *__restrict output,
                                                                               uint batch_size,
                                                                               uint num_rows,
                                                                               uint num_cols,
                                                                               uint num_rows_after_padding,
                                                                               uint num_cols_after_padding,
                                                                               uint smem_elems_per_warp,
                                                                               uint smem_rows_per_warp,
                                                                               uint output_size,
                                                                               uint num_row_steps,
                                                                               uint num_col_steps,
                                                                               uint pad) {
  uint warp_id = (threadIdx.x >> WARP_SIZE_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1);

  extern __shared__ half shmem_dynamic_half[];
  half *shmem = shmem_dynamic_half + (warp_id * smem_elems_per_warp);

  const half *sample_input = input + num_rows * num_cols * sample_id;
  for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
    for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
      (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;

  for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
    gmem_output[idx] = shmem[idx];
  }

  wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a[M_BLOCKS];
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::col_major> b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC, wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] = __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id < pad) {
    gmem_output[lane_id + output_size - 1] = __float2half(0);
  }
}

void dotBasedInteract(const void *input,
                      void *output,
                      uint batch_size,
                      uint num_rows,
                      uint num_cols,
                      uint pad,
                      bool is_fp16) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint warps_per_threadblock = 4;
  const uint threadblock_size = warps_per_threadblock * 32;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  uint num_row_steps = num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  const uint K_BLOCKS = 8;
  const uint M_BLOCKS = 2;
  const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
  // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
  const uint smem_rows_per_warp = M_BLOCKS << 4;
  const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
  const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
  const uint smem_elems_per_warp_acc = M_BLOCKS * 16 * SMEM_STRIDE_ACC * 2;  // output in FP32
  const uint smem_elems_per_warp =
      (smem_elems_per_warp_mat > smem_elems_per_warp_acc) ? smem_elems_per_warp_mat : smem_elems_per_warp_acc;
  uint output_size = num_cols + (num_rows * (num_rows - 1) >> 1) + pad;

  if (is_fp16) {
    int shmem_size = warps_per_threadblock * smem_elems_per_warp * sizeof(__half);
    dotBasedInteractFP16Kernel<warps_per_threadblock,
                               threadblock_size,
                               M_BLOCKS,
                               K_BLOCKS,
                               SMEM_STRIDE,
                               SMEM_STRIDE_ACC,
                               kWarpSize,
                               kWarpSizeLog2,
                               kTileDim,
                               kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           shmem_size>>>((const __half *) input,
                         (half *) output,
                         batch_size,
                         num_rows,
                         num_cols,
                         num_rows_after_padding,
                         num_cols_after_padding,
                         smem_elems_per_warp,
                         smem_rows_per_warp,
                         output_size,
                         num_row_steps,
                         num_col_steps,
                         pad);
  }
}

bool read_buff(const char* filename, void **hbuff, void **dbuff, size_t *bytes, bool to_device = true) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    std::cout << "Invalid file " << filename << std::endl;
    return false;
  }
  struct stat stats;

  fstat(fd, &stats);
  *bytes = stats.st_size;

  std::cout << "Read " << filename << " bytes " << *bytes << std::endl;

  *hbuff = malloc(*bytes);
  cudaMalloc(dbuff, *bytes);

  pread(fd, *hbuff, *bytes, 0);
  if (to_device) {
   cudaMemcpy(*dbuff, *hbuff, *bytes, cudaMemcpyDefault);
  }

  close(fd);
  return true;
}

bool write_buff(const char* filename, void *buff, size_t bytes) {
  int fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC);
  if (fd < 0) {
    std::cout << "Invalid file " << filename << std::endl;
    return false;
  }

  pwrite(fd, buff, bytes, 0);

  close(fd);
  return true;
}

template <typename T, uint THREADBLOCK_SIZE>
__global__ __launch_bounds__(THREADBLOCK_SIZE)
void allclose_kernel(T *a, T *b, size_t num_elm, float *abs_diff, float *rel_diff) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  size_t start = (num_elm * tid) / nthreads;
  size_t end = (num_elm * (tid + 1)) / nthreads;
  for (size_t i = start; i < end; i++) {
    float a_ = (float) a[i];
    float b_ = (float) b[i];
    abs_diff[i] = abs(a_ - b_);
    if (a_ > 0) {
      rel_diff[i] = abs_diff[i] / a_;
    }
    else {
      rel_diff[i] = 0;
    }
  }
}

template <typename T>
void allclose(void *a, void *b, size_t bytes) {
  size_t num_elm = bytes / sizeof(T);
  size_t float_bytes = num_elm * sizeof(float);
  float *habs_diff, *hrel_diff;
  float *dabs_diff, *drel_diff;

  habs_diff = (float*) malloc(float_bytes);
  hrel_diff = (float*) malloc(float_bytes);
  cudaMalloc(&dabs_diff, float_bytes);
  cudaMalloc(&drel_diff, float_bytes);

  allclose_kernel<T, 1024><<<1, 1024, 0>>>(
      (T*) a, (T*) b, num_elm, dabs_diff, drel_diff);

  cudaMemcpy(habs_diff, dabs_diff, float_bytes, cudaMemcpyDefault);
  cudaMemcpy(hrel_diff, drel_diff, float_bytes, cudaMemcpyDefault);

  float max_abs_diff = 0;
  float max_rel_diff = 0;
  size_t count = 0;
  for (size_t i = 0; i < num_elm; i++) {
    if (habs_diff[i] != 0) {
      count++;
      if (habs_diff[i] > max_abs_diff) {
        max_abs_diff = habs_diff[i];
      }
      if (hrel_diff[i] > max_rel_diff) {
        max_rel_diff = hrel_diff[i];
      }
    }
  }
  if (count == 0) {
    std::cout << "RESULT: Identical" << std::endl;
  }
  else {
    std::cout << "RESULT: Not identical" << std::endl <<
      ">>> Num non-identical elements: " << count << std::endl <<
      ">>> Max absolute diff: " << max_abs_diff << std::endl <<
      ">>> Max relative diff: " << max_rel_diff << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    std::cout << "usage: " << argv[0] << " <FP: 16 | 32>" << std::endl;
    return -1;
  }

  // Configuration
  int fp = atoi(argv[1]);
  bool is_fp16 = fp == 16;
  uint batch_size = 64;
  uint num_rows = 27;
  uint num_cols = 128;
  uint pad = 0;

  // Buffers
  void *hinput, *houtput;
  void *dinput, *doutput, *doutput_ref;
  size_t input_bytes, output_bytes;

  // Performance variables
  int warmup_iter = 5, run_iter = 10;
  float elapsed = 0;
  cudaEvent_t start, stop;

  if (!is_fp16) {
    std::cout << "FP32 is not supported in CUDA" << std::endl;
    return -1;
  }

  std::cout << "Evaluate FP " << fp << std::endl;

  std::string input_file = "data/input_fp" + std::to_string(fp);
  std::string output_file = "data/output_fp" + std::to_string(fp);

  // Read input and reference output data from file
  if (!read_buff(input_file.c_str(), &hinput, &dinput, &input_bytes) ||
      !read_buff(output_file.c_str(), &houtput, &doutput_ref, &output_bytes)) {
    return 1;
  }
  cudaMalloc(&doutput, output_bytes);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Run warm-up iterations
  for (int i = 0; i < warmup_iter; i++) {
    dotBasedInteract(dinput, doutput, batch_size, num_rows, num_cols, pad, is_fp16);
  }

  cudaDeviceSynchronize();

  // Run benchmark and time
  cudaEventRecord(start);
  for (int i = 0; i < run_iter; i++) {
    dotBasedInteract(dinput, doutput, batch_size, num_rows, num_cols, pad, is_fp16);
  }
  cudaEventRecord(stop);

  cudaDeviceSynchronize();

  // Print kernel time
  cudaEventElapsedTime(&elapsed, start, stop);
  std::cout << "Kernel time: " << std::setprecision(2) << (elapsed / run_iter) << " ms" << std::endl;

  // Test correctness
  if (is_fp16) {
    allclose<__half>(doutput_ref, doutput, output_bytes);
  }
  return 0;
}
