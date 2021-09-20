#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Common.hpp"
#include "Performance.h"
#include "Utils.h"

// The testing interface instantiates fp64 typed tests for all
// target devices. MI-100 mfma needs to be instantiated at compile time,
// but it doesn't do anything except provide a deprecation warning (e.g. not supported).
// A run-time check will abort the MI-100 fp64 tests anyway.
// Silence this warning for MmaSyncTests, as test coverage is needed
// for fp64 on all other targets which succeed MI-100.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "WMMA.h"
#pragma GCC diagnostic pop

#include <gtest/gtest.h>

#ifdef WMMA_VALIDATE_TESTS
#include "Reference.h" // Vanilla CPU kernel
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
#include "rocBLASReference.h" // rocBLAS GPU kernel
#endif // WMMA_VALIDATE_WITH_ROCBLAS
#endif // WMMA_VALIDATE_TESTS

static bool headerPrinted = false;

enum DeviceId_t : uint32_t
{
    GFX908 = 0,
    GFX90A = 1,
    UNKNOWN,
};

DeviceId_t getCurrentDeviceId()
{
    int deviceId = 0;
    CHECK_HIP_ERROR(hipGetDevice(&deviceId));
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, deviceId));
    std::string deviceName(prop.gcnArchName);

    DeviceId_t idx = DeviceId_t::UNKNOWN;

    if(deviceName.find("gfx908") != std::string::npos)
    {
        idx = DeviceId_t::GFX908;
    }
    else if(deviceName.find("gfx90a") != std::string::npos)
    {
        idx = DeviceId_t::GFX90A;
    }
    return idx;
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD>
__global__ void __launch_bounds__(256, 1) mmaSyncTestCoopLds(uint32_t       m,
                                                          uint32_t       n,
                                                          uint32_t       k,
                                                          InputT const*  a,
                                                          InputT const*  b,
                                                          OutputT const* c,
                                                          OutputT*       d,
                                                          ComputeT       alpha,
                                                          ComputeT       beta)
{
    // Setup global mapping
    using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
    using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
    using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
    using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;

    using FragA   = wmma::fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
    using FragB   = wmma::fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
    using FragC   = wmma::fragment<accumulator, BlockM, BlockN, BlockK, OutputT>;
    using FragAcc = wmma::fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

    int lda = std::is_same<LayoutA, row_major>::value ? k : m;
    int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
    int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
    int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

    // Will store to LDS as though it were a register file.
    // Rows = register count
    // Cols = unpacked register elements = 64
    // Row major to minimize bank conflicts
    constexpr uint32_t registerFileWidth = 64;
    using MappingLdsA = MappingUtil<FragA::size(), registerFileWidth, InputT, row_major>;
    using MappingLdsB = MappingUtil<FragB::size(), registerFileWidth, InputT, row_major>;
    using FragLdsA    = wmma::
        fragment<register_file_coop_a, 1, registerFileWidth, FragA::size(), InputT, row_major>;
    using FragLdsB = wmma::
        fragment<register_file_coop_b, 1, registerFileWidth, FragB::size(), InputT, row_major>;

    static_assert(FragA::size() * registerFileWidth == BlockM * BlockK,
                  "Elements of A don't match");
    static_assert(FragLdsA::size() == FragA::size(), "A Sizes don't match");
    static_assert(FragB::size() * registerFileWidth == BlockK * BlockN, "Elements don't match");
    static_assert(FragLdsB::size() == FragB::size(), "Sizes don't match");

    // Target C / D block on 2D grid
    auto matrixCoordC = MappingC::matrixCoord();

    if(std::get<0>(matrixCoordC) < m && std::get<1>(matrixCoordC) < n && BlockK < k)
    {
        // Initialize accumulator
        auto fragAcc = FragAcc();
        wmma::fill_fragment(fragAcc, static_cast<ComputeT>(0));

        // Accumulate A * B
        if(alpha)
        {
            // Setup starting addresses
            auto* addrA = MappingA::dataCoord(a, lda, std::make_pair(std::get<0>(matrixCoordC), 0));
            auto* addrB = MappingB::dataCoord(b, ldb, std::make_pair(0, std::get<1>(matrixCoordC)));

            // Prefetch the first block from global memory
            auto fragA = FragA();
            auto fragB = FragB();
            wmma::load_matrix_sync(fragA, addrA, lda);
            wmma::load_matrix_sync(fragB, addrB, ldb);

            // Setup a register file in LDS which is friendly to minimizing bank conflicts.
            // Treating register file as row_major layout with register width = 64.
            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto workgroupDim = MappingLdsA::workgroupDim();
            auto ldLds        = registerFileWidth;

            // For A, work can be shared by waves in same workgroup row because they load the same A data.
            // For B, work can be shared by waves in same workgroup col because they load the same B data.
            // E.g.
            // A blocks needed = WG.rows
            // B blocks needed = WG.cols
            // LDS layout is a register file of A blocks, followed by B blocks.
            auto* baseAddrLdsA = reinterpret_cast<InputT*>(localMemPtr);
            auto* baseAddrLdsB
                = baseAddrLdsA + std::get<0>(workgroupDim) * FragLdsA::size() * ldLds;

            auto* addrLdsA
                = baseAddrLdsA + std::get<0>(MappingLdsA::waveCoord()) * FragLdsA::size() * ldLds;
            auto* addrLdsB
                = baseAddrLdsB + std::get<1>(MappingLdsB::waveCoord()) * FragLdsB::size() * ldLds;

            wmma::store_matrix_coop_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragA), ldLds);
            wmma::store_matrix_coop_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragB), ldLds);

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x n
            auto incrA = MappingA::dataOffset(lda, std::make_pair(0, BlockK));
            auto incrB = MappingB::dataOffset(ldb, std::make_pair(BlockK, 0));

            auto endA = addrA + incrA * (k / BlockK);

            addrA += incrA;
            addrB += incrB;

            while(addrA != endA)
            {
                // Each wave must load a copy of the full fragment from LDS.
                // Synchronize here to ensure that all waves have written to LDS.
                wmma::synchronize_workgroup();
                wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
                wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);

                // Start pulling in the next block
                auto fragANext = FragA();
                auto fragBNext = FragB();
                wmma::load_matrix_sync(fragANext, addrA, lda);
                wmma::load_matrix_sync(fragBNext, addrB, ldb);

                // Mma for current block.
                wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

                // MUST synchronize here to ensure that all waves have read from LDS.
                wmma::synchronize_workgroup();
                wmma::store_matrix_coop_sync(
                    addrLdsA, reinterpret_cast<FragLdsA&>(fragANext), ldLds);
                wmma::store_matrix_coop_sync(
                    addrLdsB, reinterpret_cast<FragLdsB&>(fragBNext), ldLds);

                addrA += incrA;
                addrB += incrB;
            }

            // Sync point and mma for the last block
            wmma::synchronize_workgroup();
            wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
            wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);
            wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Load C
        auto fragC = FragC();
        wmma::fill_fragment(fragC, static_cast<OutputT>(0));
        if(beta)
        {
            // Setup address
            auto* addrC = MappingC::dataCoord(c, ldc, matrixCoordC);
            wmma::load_matrix_sync(fragC,
                                   addrC,
                                   ldc,
                                   std::is_same<LayoutC, row_major>::value ? wmma::mem_row_major
                                                                           : wmma::mem_col_major);
        }

        // D = alpha * accumAB + beta * C
#pragma unroll
        for(int i = 0; i < fragC.num_elements; ++i)
        {
            fragC.x[i] = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
        }

        // Output addresss
        auto* addrD = MappingD::dataCoord(d, ldd, matrixCoordC);

        // Store the output
        wmma::store_matrix_sync(addrD,
                                fragC,
                                ldd,
                                std::is_same<LayoutD, row_major>::value ? wmma::mem_row_major
                                                                        : wmma::mem_col_major);
    }
}

template <uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
struct Kernel
{
    public:
    Kernel(uint32_t TBlockXI,
           uint32_t TBlockYI,
           uint32_t m,
           uint32_t n,
           uint32_t k,
           ComputeT alpha,
           ComputeT beta)
    {
        TBlockX = TBlockXI;
        TBlockY = TBlockYI;
        M = m;
        N = n;
        K = k;
        Alpha = alpha;
        Beta = beta;

        // Minimum matrix sizes
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK)
        {
            return;
        }

        // Max LDS usage
        if(LDS_MAX_BYTES
            < sizeof(InputT) * (TBlockX / 64 * BlockM * BlockK + TBlockY * BlockK * BlockN))
        {
            return;
        }

        idx = getCurrentDeviceId();
        // gfx908 does not have mfma for fp64
        if(idx == DeviceId_t::UNKNOWN
            || (idx == DeviceId_t::GFX908 && std::is_same<InputT, float64_t>::value))
        {
            return;
        }

        int lda = std::is_same<LayoutA, row_major>::value ? K : M;
        int ldb = std::is_same<LayoutB, row_major>::value ? N : K;
        int ldc = std::is_same<LayoutC, row_major>::value ? N : M;
        int ldd = std::is_same<LayoutD, row_major>::value ? N : M;

        if(!headerPrinted)
        {
            std::cout
                << "TBlkX, TBlkY, BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, "
                "ldd, LytA_LytB_LytC_LytD, Ti_To_Tc, elapsedMs, GFlops, GFlops/s, Efficiency(%)\n";
            headerPrinted = true;
        }

        std::cout << TBlockX << ", " << TBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
                  << ", " << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
                  << ", " << beta << ", " << ldc << ", " << ldd << ", "
                  << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << "_"
                  << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << "_"
                  << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << "_"
                  << (std::is_same<LayoutD, row_major>::value ? "R" : "C") << ", "
                  << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
                  << dataTypeToString<ComputeT>();

        if(quirks::hipcc_bug_half_packing<OutputT, LayoutC>::value)
        {
            std::cout << ", "
                      << "na"
                      << ", "
                      << "na"
                      << ", "
                      << "na"
                      << ", "
                      << "na"
                      << ", "
                      << " SKIPPED" << std::endl;
            return;
        }

        // Initialize input matrices
        matrixA.resize(M * K);
        matrixB.resize(K * N);
        matrixC.resize(M * N, OutputT(0));
        matrixD.resize(M * N);

        MatrixUtil<LayoutA>::fill(matrixA, M, K);
        MatrixUtil<LayoutB>::fill(matrixB, K, N);
        MatrixUtil<LayoutC>::fill(matrixC, M, N);
        MatrixUtil<LayoutD>::fill(matrixD, M, N, std::numeric_limits<OutputT>::signaling_NaN());

        const size_t bytesA = matrixA.size() * sizeof(InputT);
        const size_t bytesB = matrixB.size() * sizeof(InputT);
        const size_t bytesC = matrixC.size() * sizeof(OutputT);
        const size_t bytesD = matrixD.size() * sizeof(OutputT);

        CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
        CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
        CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
        CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

        CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

        gridDim
        = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

        blockDim = dim3(TBlockX, TBlockY);
    }

    void MmaSyncTestCoopLdsKernel()
    {
        // Minimum matrix sizes
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK)
        {
            return;
        }

        // Max LDS usage
        if(LDS_MAX_BYTES
            < sizeof(InputT) * (TBlockX / 64 * BlockM * BlockK + TBlockY * BlockK * BlockN))
        {
            return;
        }
    
        auto idx = getCurrentDeviceId();
        // gfx908 does not have mfma for fp64
        if(idx == DeviceId_t::UNKNOWN
            || (idx == DeviceId_t::GFX908 && std::is_same<InputT, float64_t>::value))
        {
            return;
        }

        if(quirks::hipcc_bug_half_packing<OutputT, LayoutC>::value)
        {
            return;
        }

        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

        hipExtLaunchKernelGGL(
            (mmaSyncTestCoopLds<BlockM,
                             BlockN,
                             BlockK,
                             InputT,
                             OutputT,
                             ComputeT,
                             LayoutA,
                             LayoutB,
                             LayoutC,
                             LayoutD>),
            gridDim,
            blockDim,
            sizeof(InputT)
                * (blockDim.x / 64 * BlockM * BlockK + blockDim.y * BlockK * BlockN), // sharedMemBytes
            0, // stream
            startEvent, // Event start
            stopEvent, // event stop
            0, // flags
            M,
            N,
            K,
            d_a,
            d_b,
            d_c,
            d_d,
            Alpha,
            Beta);

        auto elapsedTimeMs = 0.0f;
        CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
        CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
        CHECK_HIP_ERROR(hipEventDestroy(startEvent));
        CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

        auto totalGFlops      = calculateGFlops(M, N, K);
        auto peakGFlopsPerSec = (idx == DeviceId_t::GFX908)
                                    ? calculatePeakGFlopsPerSec<InputT, MI100>(1087)
                                    : calculatePeakGFlopsPerSec<InputT, MI200>(985);

        auto actualGFlopsPerSec = calculateGFlopsPerSec(M, N, K, elapsedTimeMs);
        auto efficiency         = actualGFlopsPerSec / peakGFlopsPerSec * 100.0f;

        std::cout << ", " << elapsedTimeMs << ", " << totalGFlops << ", " << actualGFlopsPerSec << ", "
              << efficiency << ", ";
    }

    ~Kernel()
    {
        // Minimum matrix sizes
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK)
        {
            return;
        }

        // Max LDS usage
        if(LDS_MAX_BYTES
            < sizeof(InputT) * (TBlockX / 64 * BlockM * BlockK + TBlockY * BlockK * BlockN))
        {
            return;
        }

        // gfx908 does not have mfma for fp64
        if(idx == DeviceId_t::UNKNOWN
            || (idx == DeviceId_t::GFX908 && std::is_same<InputT, float64_t>::value))
        {
            return;
        }

        if(quirks::hipcc_bug_half_packing<OutputT, LayoutC>::value)
        {
            return;
        }

#ifdef WMMA_VALIDATE_TESTS
        const size_t bytesD = matrixD.size() * sizeof(OutputT);
        CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

        // Init reference data and then validate
        std::vector<OutputT> matrixD_ref(M * N, OutputT(0));

        // Give more error tolerance to ComputeT = fp16,
        // due to MFMA output is always fp32. We downcast the MFMA result to fp16, which
        // will introduce an error compared to native fp16 MAC. The tolerance would be a function
        // of max / min values and number of operations propagating the error.
        // Note that integer values between [-2048, 2048 ] are exactly representable by fp16,
        // and significant rounding errors occur thereafter to the nearest multiple of 2.
        // The input generator for GEMM uses integer values within a certain range, therefore
        // FMA operations will be very prone to significant errors.
        double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;

        bool validated = false;
#ifdef WMMA_VALIDATE_WITH_ROCBLAS
        if(quirks::rocblas_supported<InputT, OutputT, ComputeT>::value)
        {
            // rocblas matrix C, D always in col_major
            MatrixUtil<col_major>::fill(matrixC, M, N);
            gemm_rocBLAS<InputT, OutputT, ComputeT, LayoutA, LayoutB>(M,
                                                                      N,
                                                                      K,
                                                                      matrixA.data(),
                                                                      matrixB.data(),
                                                                      matrixC.data(),
                                                                      matrixD_ref.data(),
                                                                      Alpha,
                                                                      Beta);

            EXPECT_TRUE((compareEqual<OutputT, OutputT, LayoutD, col_major>(
                matrixD, matrixD_ref, M, N, errorTolerance)));


            validated = true;
        }
#endif // WMMA_VALIDATE_WITH_ROCBLAS
        if(!validated)
        {
            gemm_CPU<InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD>(M,
                                                                                    N,
                                                                                    K,
                                                                                    matrixA.data(),
                                                                                    matrixB.data(),
                                                                                    matrixC.data(),
                                                                                    matrixD_ref.data(),
                                                                                    Alpha,
                                                                                    Beta);
            EXPECT_TRUE((compareEqual<OutputT, OutputT, LayoutD, LayoutD>(
                matrixD, matrixD_ref, M, N, errorTolerance)));

        }

#else // WMMA_VALIDATE_TESTS
    // No validation, close off the line.
        std::cout << std::endl;

#endif // WMMA_VALIDATE_TESTS

    // Release device memory
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
    }
    private:
        uint32_t M, N, K, TBlockX, TBlockY;
        ComputeT Alpha, Beta;
        InputT *d_a, *d_b;
        OutputT *d_c, *d_d;
        dim3 gridDim, blockDim;
        std::vector<InputT>  matrixA, matrixB;
        std::vector<OutputT> matrixC, matrixD;
        DeviceId_t idx;        
};

template <typename T>
struct MmaSyncTestCoopLdsWrapper;

template<typename BlockM, typename BlockN, typename BlockK, 
         typename InputT, typename OutputT, typename ComputeT>
struct MmaSyncTestCoopLdsWrapper<std::tuple<BlockM, BlockN, BlockK, InputT, OutputT, ComputeT>> : public testing::Test
{
    void SetUp() override 
    {
        // clang-format off
        std::vector<std::array<int, 2>> thread_block = {{64, 1}, {64, 2}, {64, 4},
                                                        {128,1}, {128,2},
                                                        {256,1}};

        std::vector<std::array<int, 3>> problem_sizes = {{64, 64, 1024},
                                                         {32, 64, 1024},
                                                         {64, 32, 1024},
                                                         {256, 256, 1024},
                                                         {2048, 64, 1024},
                                                         {64, 2048, 1024},
                                                         {1024, 1024, 1024},
                                                         {2048, 2048, 2048},
                                                         {2560, 2560, 2560},
                                                         {3072, 3072, 3072},
                                                         {3584, 3584, 3584},
                                                         {4096, 4096, 4096},
                                                         {5120, 5120, 5120},
                                                         {6144, 6144, 6144},
                                                         {7168, 7168, 7168},
                                                         {8192, 8192, 8192}};
    
        // clang-format on
        int i = 0;
        for(auto tblock : thread_block)
        {
            for(auto size : problem_sizes)
            {
                // skip large sizes when in validation mode
                #ifdef WMMA_VALIDATE_TESTS
                if(size[0] * size[1] > 1024 * 1024)
                    continue;
                #endif // WMMA_VALIDATE_TESTS
                std::tuple<row_major, col_major> types;
                for_each(types, [&](auto layout_a) {
                    for_each(types, [&](auto layout_b) {
                        for_each(types, [&](auto layout_c) {
                            Kernel<BlockM::value, BlockN::value, BlockK::value, InputT,
                                OutputT, ComputeT, decltype(layout_a), decltype(layout_b),
                                decltype(layout_c), decltype(layout_c)> *obj = new Kernel<BlockM::value, BlockN::value, BlockK::value, InputT,
                                OutputT, ComputeT, decltype(layout_a), decltype(layout_b),
                                decltype(layout_c), decltype(layout_c)>(tblock[0], tblock[1], size[0], size[1], size[2], ComputeT(2.0f), ComputeT(2.0f));

                            obj->MmaSyncTestCoopLdsKernel();

                            delete obj;
                        });
                    });
                });
            }
        }
    }

    void TearDown() override
    {
    }
};

using Implementations = testing::Types<
    // InputT, OutputT, ComputeT
    // Non-native bfloat16_t
    std::tuple<I<16>, I<16>, I<16>, bfloat16_t, bfloat16_t, bfloat16_t>, std::tuple<I<16>, I<16>, I<32>, bfloat16_t, bfloat16_t, bfloat16_t>,
    std::tuple<I<16>, I<16>, I<64>, bfloat16_t, bfloat16_t, bfloat16_t>, std::tuple<I<16>, I<16>, I<128>, bfloat16_t, bfloat16_t, bfloat16_t>,
    std::tuple<I<16>, I<16>, I<256>, bfloat16_t, bfloat16_t, bfloat16_t>, std::tuple<I<32>, I<32>, I<8>, bfloat16_t, bfloat16_t, bfloat16_t>,
    std::tuple<I<32>, I<32>, I<16>, bfloat16_t, bfloat16_t, bfloat16_t>, std::tuple<I<32>, I<32>, I<32>, bfloat16_t, bfloat16_t, bfloat16_t>,
    std::tuple<I<32>, I<32>, I<64>, bfloat16_t, bfloat16_t, bfloat16_t>, std::tuple<I<32>, I<32>, I<128>, bfloat16_t, bfloat16_t, bfloat16_t>,

    
    std::tuple<I<16>, I<16>, I<16>, bfloat16_t, bfloat16_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, bfloat16_t, bfloat16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, bfloat16_t, bfloat16_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, bfloat16_t, bfloat16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, bfloat16_t, bfloat16_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, bfloat16_t, bfloat16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, bfloat16_t, bfloat16_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, bfloat16_t, bfloat16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, bfloat16_t, bfloat16_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, bfloat16_t, bfloat16_t, float32_t>,

    std::tuple<I<16>, I<16>, I<16>, bfloat16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, bfloat16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, bfloat16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, bfloat16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, bfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, bfloat16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, bfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, bfloat16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, bfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, bfloat16_t, float32_t, float32_t>,

    // Native fp16
    std::tuple<I<16>, I<16>, I<16>, float16_t, float16_t, float16_t>, std::tuple<I<16>, I<16>, I<32>, float16_t, float16_t, float16_t>,
    std::tuple<I<16>, I<16>, I<64>, float16_t, float16_t, float16_t>, std::tuple<I<16>, I<16>, I<128>, float16_t, float16_t, float16_t>,
    std::tuple<I<16>, I<16>, I<256>, float16_t, float16_t, float16_t>, std::tuple<I<32>, I<32>,I<8>, float16_t, float16_t, float16_t>, 
    std::tuple<I<32>, I<32>, I<16>, float16_t, float16_t, float16_t>, std::tuple<I<32>, I<32>, I<32>, float16_t, float16_t, float16_t>,
    std::tuple<I<32>, I<32>, I<64>, float16_t, float16_t, float16_t>, std::tuple<I<32>, I<32>, I<128>, float16_t, float16_t, float16_t>,

    std::tuple<I<16>, I<16>, I<16>, float16_t, float16_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, float16_t, float16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, float16_t, float16_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, float16_t, float16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, float16_t, float16_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, float16_t, float16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, float16_t, float16_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, float16_t, float16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, float16_t, float16_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, float16_t, float16_t, float32_t>,

    std::tuple<I<16>, I<16>, I<16>, float16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, float16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, float16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, float16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, float16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, float16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, float16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, float16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, float16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, float16_t, float32_t, float32_t>,

    // Native fp32
    std::tuple<I<16>, I<16>, I<16>, float32_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, float32_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, float32_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, float32_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, float32_t, float32_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, float32_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, float32_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, float32_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, float32_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, float32_t, float32_t, float32_t>,

    // Non-native hfloat16_t (i.e. __half)
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, hfloat16_t, hfloat16_t>, std::tuple<I<16>, I<16>, I<32>, hfloat16_t, hfloat16_t, hfloat16_t>,
    std::tuple<I<16>, I<16>, I<64>, hfloat16_t, hfloat16_t, hfloat16_t>, std::tuple<I<16>, I<16>, I<128>, hfloat16_t, hfloat16_t, hfloat16_t>,
    std::tuple<I<16>, I<16>, I<256>, hfloat16_t, hfloat16_t, hfloat16_t>, std::tuple<I<32>, I<32>,I<8>, hfloat16_t, hfloat16_t, hfloat16_t>,
    std::tuple<I<32>, I<32>, I<16>, hfloat16_t, hfloat16_t, hfloat16_t>, std::tuple<I<32>, I<32>, I<32>, hfloat16_t, hfloat16_t, hfloat16_t>,
    std::tuple<I<32>, I<32>, I<64>, hfloat16_t, hfloat16_t, hfloat16_t>, std::tuple<I<32>, I<32>, I<128>, hfloat16_t, hfloat16_t, hfloat16_t>,

    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, hfloat16_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, hfloat16_t, hfloat16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, hfloat16_t, hfloat16_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, hfloat16_t, hfloat16_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, hfloat16_t, hfloat16_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, hfloat16_t, hfloat16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, hfloat16_t, hfloat16_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, hfloat16_t, hfloat16_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, hfloat16_t, hfloat16_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, hfloat16_t, hfloat16_t, float32_t>,

    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<32>, hfloat16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<64>, hfloat16_t, float32_t, float32_t>, std::tuple<I<16>, I<16>, I<128>, hfloat16_t, float32_t, float32_t>,
    std::tuple<I<16>, I<16>, I<256>, hfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>,I<8>, hfloat16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<16>, hfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<32>, hfloat16_t, float32_t, float32_t>,
    std::tuple<I<32>, I<32>, I<64>, hfloat16_t, float32_t, float32_t>, std::tuple<I<32>, I<32>, I<128>, hfloat16_t, float32_t, float32_t>,

    // Native int8
    std::tuple<I<16>, I<16>, I<16>, int8_t, int32_t, int32_t>, std::tuple<I<16>, I<16>, I<32>, int8_t, int32_t, int32_t>,
    std::tuple<I<16>, I<16>, I<64>, int8_t, int32_t, int32_t>, std::tuple<I<16>, I<16>, I<128>, int8_t, int32_t, int32_t>,
    std::tuple<I<16>, I<16>, I<256>, int8_t, int32_t, int32_t>, std::tuple<I<32>, I<32>,I<8>, int8_t, int32_t, int32_t>,
    std::tuple<I<32>, I<32>, I<16>, int8_t, int32_t, int32_t>,  std::tuple<I<32>, I<32>, I<32>, int8_t, int32_t, int32_t>,
    std::tuple<I<32>, I<32>, I<64>, int8_t, int32_t, int32_t>,  std::tuple<I<32>, I<32>, I<128>, int8_t, int32_t, int32_t>,

    std::tuple<I<16>, I<16>, I<16>, int8_t, int8_t, int32_t>, std::tuple<I<16>, I<16>, I<32>, int8_t, int8_t, int32_t>,
    std::tuple<I<16>, I<16>, I<64>, int8_t, int8_t, int32_t>, std::tuple<I<16>, I<16>, I<128>, int8_t, int8_t, int32_t>,
    std::tuple<I<16>, I<16>, I<256>, int8_t, int8_t, int32_t>, std::tuple<I<32>, I<32>,I<8>, int8_t, int8_t, int32_t>,
    std::tuple<I<32>, I<32>, I<16>, int8_t, int8_t, int32_t>, std::tuple<I<32>, I<32>, I<32>, int8_t, int8_t, int32_t>,
    std::tuple<I<32>, I<32>, I<64>, int8_t, int8_t, int32_t>, std::tuple<I<32>, I<32>, I<128>, int8_t, int8_t, int32_t>,
    
    // float64_t
    std::tuple<I<16>, I<16>, I<16>, float64_t, float64_t, float64_t>, std::tuple<I<16>, I<16>, I<32>, float64_t, float64_t, float64_t>,
    std::tuple<I<16>, I<16>, I<64>, float64_t, float64_t, float64_t>, std::tuple<I<16>, I<16>, I<128>, float64_t, float64_t, float64_t>,
    std::tuple<I<16>, I<16>, I<256>, float64_t, float64_t, float64_t>>;

TYPED_TEST_SUITE(MmaSyncTestCoopLdsWrapper, Implementations);

TYPED_TEST(MmaSyncTestCoopLdsWrapper, mmasynctestcooplds)
{
} 
