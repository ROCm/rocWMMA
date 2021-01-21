#include <hip/hip_runtime.h>

#include "Types.h"
#include "BufferLoad.h"
#include "BufferStore.h"
#include "Utils.h"

// Remove later
#include "BufferDescriptor.h"

struct TestParams
{
    enum : uint32_t
    {
        // Matrix geometry
        BLOCK_M = 32,
        BLOCK_N = 32,
        BLOCK_K = 2,

        MAT_M = 128,
        MAT_N = 128,
        MAT_K = 128,

        // Thread counts
        BLOCK_DIM_X = 128,
        BLOCK_DIM_Y = 1,

        // Block counts
        GRID_DIM_X = 1,
        GRID_DIM_Y = 1,   
    };

    using LAYOUT = row_major; 
};

__global__ void loadTest(const float32_t* mat, float32_t* result, float32_t ldm)
{
    using Params = TestParams;
    using Loader = amdgcn_buffer_load_dword_MxNxK<
        matrix_a,
        Params::BLOCK_M,
        Params::BLOCK_N,
        Params::BLOCK_K,
        float32_t,
        row_major>;

    using Traits = typename Loader::Traits;
    
    // Move the data origin to the start of the block data.
    uint32_t startOffset = 
        (blockIdx.y * ldm * Params::BLOCK_M) +             // Start row
         blockIdx.x * (blockDim.x / 64) * Params::BLOCK_N ; // Start col

    auto loadedA = Loader::exec(
        mat + startOffset,
        std::is_same<Params::LAYOUT, row_major>::value ? Params::MAT_N : Params::MAT_M);

    uint32_t startOffsetC = 
            (blockIdx.x) * 64 * 2;

    BufferDescriptor<float> cSRD(result + startOffsetC, 64);

    for(uint32_t i = 0; i < Traits::LoadCount; i++)
    {
        __llvm_amdgcn_buffer_store_f32((loadedA[i]),
                                        *cSRD,
                                        (threadIdx.x / 64)*Traits::LoadCount,
                                        (i*64 + threadIdx.x % 64)*sizeof(float),
                                        false,
                                        false);
    }
}

template<
    typename T,
    size_t BlockM = 0,
    size_t BlockN = 0,
    size_t BlockK = 0,
    size_t TBlockX = 0, // Launch param thread block size
    size_t TBlockY = 0> // Launch param thread block size
struct BlockMetaData
{
    static constexpr size_t threads_per_wave = 64;
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
    static constexpr auto gridDim(size_t M, size_t N, size_t K) -> std::tuple<size_t, size_t, size_t>
    {
        return std::tuple<size_t, size_t, size_t>(
            ceilDiv(M, BlockM * TBlockX / threads_per_wave),
            ceilDiv(N, BlockN * TBlockY),
            ceilDiv(K, BlockK));
    }

};

int main()
{
    std::cout << "HIP vector load example\n";

    
    const int rows = 128;
    const int cols = 128;
    const int N = rows*cols;

    std::vector<float> vala(N);
    MatrixUtil<row_major>::fill(vala, rows, cols);
    // MatrixUtil<row_major>::print(vala, rows, cols);

    std::vector<float> valb(N);
    MatrixUtil<col_major>::fill(valb, rows, cols);
    // MatrixUtil<col_major>::print(valb, rows, cols);

    std::vector<float> result(N, 0.0f);

    // //validateC<32,32,32>(vala, valb, result);

    const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

    float* d_a;
    assert(hipMalloc(&d_a, valbytes) == hipSuccess);
    assert(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    float* d_b;
    assert(hipMalloc(&d_b, valbytes) == hipSuccess);
    assert(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) == hipSuccess);

    float* d_c;
    assert(hipMalloc(&d_c, valbytes) == hipSuccess); 

    int blockSize = 128;
    int blocks    = ceilDiv(N, 64*32);
    hipLaunchKernelGGL(loadTest,
                       dim3(1, 1),
                       dim3(blockSize),
                       0, // sharedMemBytes
                       0, // stream
                       d_a,
                       d_c,
                       TestParams::BLOCK_K);

    assert(hipMemcpy(result.data(), d_c, valbytes, hipMemcpyDeviceToHost) == hipSuccess);

    MatrixUtil<row_major>::print(result, 64, 8);

    // Release device memory
    assert(hipFree(d_a) == hipSuccess);
    assert(hipFree(d_b) == hipSuccess);
    assert(hipFree(d_c) == hipSuccess);

    return 0;
}