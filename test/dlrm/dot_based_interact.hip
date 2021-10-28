#include "common.h"
#include "dlrm_dot_based_interact.hip"

template <typename Func>
void run(Func func, int warmup_iter, int run_iter) {
  float elapsed = 0;
  hipEvent_t start, stop;

  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Run warm-up iterations
  for (int i = 0; i < warmup_iter; i++) {
    func();
  }

  hipDeviceSynchronize();

  // Run benchmark and time
  hipEventRecord(start);
  for (int i = 0; i < run_iter; i++) {
    func();
  }
  hipEventRecord(stop);

  hipDeviceSynchronize();

  // Print kernel time
  hipEventElapsedTime(&elapsed, start, stop);
  std::cout << "Kernel time: " << std::setprecision(2) << (elapsed / run_iter) << " ms" << std::endl;

}

int main(int argc, char **argv) {
  // Configuration
  int fp = atoi(argv[1]);
  int tile_size = 16;
  bool is_bwd = false;

  std::map<int, CmdOption> options({
      {'h', {nullptr, HELP_OPT, "help", no_argument, "Print usage"}},
      {0, {&fp, INT_OPT, "fp", required_argument,
          "Set floating point precision [options: 16, 32]"}},
      {'t', {&tile_size, INT_OPT, "tile_size", required_argument,
          "Set tile size [options: 16, 32]"}},
      {1, {&is_bwd, BOOL_OPT, "bwd", no_argument,
          "Run the backward pass instead of the forward pass"}}
      });

  get_options(argc, argv, options);

  bool is_fp16 = fp == 16;
  uint batch_size = 64;
  uint num_rows = 27;
  uint num_cols = 128;
  uint pad = 0;

  // Performance variables
  int warmup_iter = 5, run_iter = 10;
  std::string fp_str = std::to_string(fp);

  std::cout << "Mode: " << (is_bwd ? "backward" : "forward") << std::endl;
  std::cout << "FP: " << fp << std::endl;
  std::cout << "Tile size: " << tile_size << std::endl;

  if (!is_bwd) { // Run forward pass
    // Buffers
    struct buff_t input, output_ref;
    void *output;

    input.file = "dlrmData/input_fp" + fp_str;
    output_ref.file = "dlrmData/output_fp" + fp_str;

    // Read input and reference output data from file
    if (!read_buff(&input) ||
        !read_buff(&output_ref)) {
      return 1;
    }

    hipMalloc(&output, output_ref.bytes);

    run([&]() {
        dotBasedInteractFwd(
            input.d,
            nullptr,
            output,
            batch_size,
            num_rows,
            num_cols,
            pad,
            is_fp16);
        }, warmup_iter, run_iter);

    // Test correctness
    if (is_fp16) {
      allclose<half>(output_ref.d, output, output_ref.bytes);
    }
    else {
      allclose<float>(output_ref.d, output, output_ref.bytes);
    }
  }
  else { // Run backward pass
    // Buffers
    struct buff_t input, upstream_grad;
    void *grad, *bottom_mlp_grad;
    struct buff_t grad_ref, bottom_mlp_grad_ref;

    input.file = "dlrmData/input_fp" + fp_str;
    upstream_grad.file = "dlrmData/input_grad_fp" + fp_str;
    grad_ref.file = "dlrmData/output_input_grad_fp" + fp_str;
    bottom_mlp_grad_ref.file = "dlrmData/output_mlp_input_grad_fp" + fp_str;

    // Read input and reference output data from file
    if (!read_buff(&input) ||
        !read_buff(&upstream_grad) ||
        !read_buff(&grad_ref) ||
        !read_buff(&bottom_mlp_grad_ref)) {
      return 1;
    }

    hipMalloc(&grad, grad_ref.bytes);
    hipMalloc(&bottom_mlp_grad, bottom_mlp_grad_ref.bytes);

    if (tile_size == 16) {
      run([&]() {
          dotBasedInteractBwd<16>(
              input.d,
              upstream_grad.d,
              grad,
              bottom_mlp_grad,
              batch_size,
              num_rows,
              num_cols,
              pad,
              is_fp16);
          }, warmup_iter, run_iter);
    }
    else {
      run([&]() {
          dotBasedInteractBwd<32>(
              input.d,
              upstream_grad.d,
              grad,
              bottom_mlp_grad,
              batch_size,
              num_rows,
              num_cols,
              pad,
              is_fp16);
          }, warmup_iter, run_iter);
    }

    if (is_fp16) {
      std::cout << "Validate grad" << std::endl;
      allclose<half>(grad_ref.d, grad, grad_ref.bytes);
      std::cout << "Validate bottom_mlp_grad" << std::endl;
      allclose<half>(bottom_mlp_grad_ref.d, bottom_mlp_grad, bottom_mlp_grad_ref.bytes);
    }
    else {
      std::cout << "Validate grad" << std::endl;
      allclose<float>(grad_ref.d, grad, grad_ref.bytes);
      std::cout << "Validate bottom_mlp_grad" << std::endl;
      allclose<float>(bottom_mlp_grad_ref.d, bottom_mlp_grad, bottom_mlp_grad_ref.bytes);
    }
  }
  return 0;
}
