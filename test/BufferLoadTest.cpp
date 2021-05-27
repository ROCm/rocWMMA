#include <hip/hip_runtime.h>

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "Types.h"
#include "Utils.h"

#include "WMMA.h"

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
    using MATRIX = accumulator;
};

__global__ void loadTest(const float32_t* mat, float32_t* result, uint32_t ldm, uint32_t ldr)
{
    using Params = TestParams;
    using Loader = amdgcn_buffer_load_dword_DxK<accumulator,
                                                Params::BLOCK_N,
                                                Params::BLOCK_K,
                                                Params::TYPE,
                                                Params::LAYOUT>;

    using Traits = typename Loader::Traits;

    using MappingUtil = MappingUtil<Params::TYPE, Params::BLOCK_M, Params::BLOCK_N, Params::LAYOUT>;

    // Move the data origin to the start of the block data.
    auto* blockAData = MappingUtil::dataCoordN0(mat, ldm);
    auto* blockBData = MappingUtil::dataCoordM0(mat, ldm);
    auto* blockCData = MappingUtil::dataCoord(mat, ldm);

    auto fragA = wmma::fragment<matrix_a,
                                Params::BLOCK_M,
                                Params::BLOCK_N,
                                Params::BLOCK_K,
                                Params::TYPE,
                                row_major>();

    auto fragB = wmma::fragment<matrix_b,
                                Params::BLOCK_M,
                                Params::BLOCK_N,
                                Params::BLOCK_K,
                                Params::TYPE,
                                row_major>();

    auto fragC = wmma::
        fragment<accumulator, Params::BLOCK_M, Params::BLOCK_N, Params::BLOCK_K, Params::TYPE>();

    wmma::load_matrix_sync(fragA, blockAData, ldm);
    wmma::load_matrix_sync(fragB, blockBData, ldm);

    wmma::mma_sync(fragC, fragA, fragB, fragC);
    //Loader::exec(blockAddr, ldm);

    // uint32_t startOffsetC = (blockIdx.x + blockIdx.y * gridDim.x) * // Initial index
    //                         (*fragA).size() * blockDim.y * // Number of regs
    //                         (blockDim.x / Params::WAVE_SIZE) * // Blocks per wave
    //                         ldr; // Register size of 64 elements

    // BufferDescriptor<float> srd(result + startOffsetC, ldr); // Register file

    // for(uint32_t i = 0; i < Traits::LoadCount; i++) // Write my registers
    // {
    //     __llvm_amdgcn_buffer_store_f32(
    //         (*fragA)[i],
    //         *srd,
    //         ((threadIdx.x + threadIdx.y * blockDim.x) / ldr) * Traits::LoadCount,
    //         (i * ldr + ((threadIdx.x + threadIdx.y * blockDim.x) % ldr)) * sizeof(float),
    //         false,
    //         false);
    // }
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
    CHECK_HIP_ERROR(hipMalloc(&d_a, valbytes));
    CHECK_HIP_ERROR(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice));

    TestParams::TYPE* d_r;
    CHECK_HIP_ERROR(hipMalloc(&d_r, valbytes));

    hipLaunchKernelGGL(loadTest,
                       dim3(TestParams::GRID_DIM_X, TestParams::GRID_DIM_Y),
                       dim3(TestParams::BLOCK_DIM_X, TestParams::BLOCK_DIM_Y),
                       0, // sharedMemBytes
                       0, // stream
                       d_a,
                       d_r,
                       ldm,
                       64);

    CHECK_HIP_ERROR(hipMemcpy(result.data(), d_r, valbytes, hipMemcpyDeviceToHost));

    MatrixUtil<row_major>::print(result, 64, 64);

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_r));

    return 0;
}
