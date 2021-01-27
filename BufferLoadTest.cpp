#include <hip/hip_runtime.h>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "Types.h"
#include "Utils.h"

// Remove later
#include "BufferDescriptor.h"

struct TestParams
{
    enum : uint32_t
    {
        // Matrix geometry
        MAT_M = 128,
        MAT_N = 128,
        MAT_K = 128,

        // Block geometry
        BLOCK_M = 32,
        BLOCK_N = 32,
        BLOCK_K = 32,

        // Thread counts
        BLOCK_DIM_X = 128,
        BLOCK_DIM_Y = 2,

        // Block counts
        GRID_DIM_X = 2,
        GRID_DIM_Y = 2,

        // WAVE_SIZE
        WAVE_SIZE = AMDGCN_WAVE_SIZE,
    };

    using LAYOUT = row_major;
    using TYPE   = float32_t;
};

__global__ void loadTest(const float32_t* mat, float32_t* result, uint32_t ldm, uint32_t ldr)
{
    using Params = TestParams;
    using Loader = amdgcn_buffer_load_dword_DxK<matrix_b,
                                                Params::BLOCK_N,
                                                Params::BLOCK_K,
                                                Params::TYPE,
                                                Params::LAYOUT>;

    using Traits = typename Loader::Traits;

    using MappingUtil = MappingUtil<Params::TYPE, Params::BLOCK_M, Params::BLOCK_N, Params::LAYOUT>;

    // Move the data origin to the start of the block data.
    auto* blockAddr = MappingUtil::dataCoord(mat, ldm);
    auto  loadedA   = Loader::exec(blockAddr, ldm);

    uint32_t startOffsetC = (blockIdx.x + blockIdx.y * gridDim.x) * // Initial index
                            loadedA.size() * blockDim.y * // Number of regs
                            (blockDim.x / Params::WAVE_SIZE) * // Blocks per wave
                            ldr; // Register size of 64 elements

    BufferDescriptor<float> srd(result + startOffsetC, ldr); // Register file

    for(uint32_t i = 0; i < Traits::LoadCount; i++) // Write my registers
    {
        __llvm_amdgcn_buffer_store_f32(
            loadedA[i],
            *srd,
            ((threadIdx.x + threadIdx.y * blockDim.x) / ldr) * Traits::LoadCount,
            (i * ldr + ((threadIdx.x + threadIdx.y * blockDim.x) % ldr)) * sizeof(float),
            false,
            false);
    }
}

template <typename T,
          size_t BlockM  = 0,
          size_t BlockN  = 0,
          size_t BlockK  = 0,
          size_t TBlockX = 0, // Launch param thread block size
          size_t TBlockY = 0> // Launch param thread block size
struct BlockMetaData
{
    static constexpr size_t threads_per_wave  = 64;
    static constexpr size_t elements_per_vgpr = 256 / sizeof(T);

    static constexpr auto blockStrides() -> std::tuple<size_t, size_t, size_t>
    {
        return std::tuple<size_t, size_t, size_t>(BlockM, BlockN, BlockK);
    }

    static constexpr auto blockLaunchDims() -> std::tuple<size_t, size_t>
    {
        return std::tuple<size_t, size_t>(TBlockX, TBlockY);
    }

    // How many mxnxk blocks total.
    static constexpr auto gridDim(size_t M, size_t N, size_t K)
        -> std::tuple<size_t, size_t, size_t>
    {
        return std::tuple<size_t, size_t, size_t>(ceilDiv(M, BlockM * TBlockX / threads_per_wave),
                                                  ceilDiv(N, BlockN * TBlockY),
                                                  ceilDiv(K, BlockK));
    }
};

int main()
{
    std::cout << "HIP vector load example\n";

    const int rows = TestParams::MAT_M;
    const int cols = TestParams::MAT_K;
    const int N    = rows * cols;
    const int ldm  = std::is_same<TestParams::LAYOUT, row_major>::value ? cols : rows;

    std::vector<TestParams::TYPE> vala(N);
    MatrixUtil<TestParams::LAYOUT>::fill(vala, rows, cols);
    // MatrixUtil<row_major>::print(vala, rows, cols);

    std::vector<TestParams::TYPE> result(N, 0.0f);

    const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

    TestParams::TYPE* d_a;
    assert(hipMalloc(&d_a, valbytes) == hipSuccess);
    assert(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    TestParams::TYPE* d_r;
    assert(hipMalloc(&d_r, valbytes) == hipSuccess);

    hipLaunchKernelGGL(loadTest,
                       dim3(TestParams::GRID_DIM_X, TestParams::GRID_DIM_Y),
                       dim3(TestParams::BLOCK_DIM_X, TestParams::BLOCK_DIM_Y),
                       0, // sharedMemBytes
                       0, // stream
                       d_a,
                       d_r,
                       ldm,
                       64);

    assert(hipMemcpy(result.data(), d_r, valbytes, hipMemcpyDeviceToHost) == hipSuccess);

    MatrixUtil<row_major>::print(result, 64, 32);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_r) == hipSuccess);

    return 0;
}
