#include <hip/hip_runtime.h>

#include <type_traits>
#include <unistd.h>
#include <random>
#include <utility>
#include "Constants.h"
#include "Types.h"
#include "Utils.h"
#include "Reference.h"

#include "WMMA.h"
#include <gtest/gtest.h>

#include "Common.hpp"
#define NO_OF_TESTS 98

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
__global__ void barrierTest(uint32_t       m,
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
    using FragLdsA
        = wmma::fragment<register_file, 1, registerFileWidth, FragA::size(), InputT, row_major>;
    using FragLdsB
        = wmma::fragment<register_file, 1, registerFileWidth, FragB::size(), InputT, row_major>;

    static_assert(FragA::size() * 64 == BlockM * BlockK, "Elements of A don't match");
    static_assert(FragLdsA::size() == FragA::size(), "A Sizes don't match");
    static_assert(FragB::size() * 64 == BlockK * BlockN, "Elements don't match");
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
            // Register file blocks in LDS follow same wg mapping for convenience.
            // Each wave will prefetch one block of A and one block of B.
            // A blocks occupy first portion of LDS and B blocks occupy the latter.
            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto workgroupDim = MappingLdsA::workgroupDim();
            auto ldLds        = registerFileWidth * std::get<1>(workgroupDim);

            auto* baseAddrLdsA = reinterpret_cast<InputT*>(localMemPtr);
            auto* baseAddrLdsB
                = baseAddrLdsA + std::get<0>(workgroupDim) * FragLdsA::size() * ldLds;

            auto matrixCoordLdsA = MappingLdsA::matrixCoord(MappingLdsA::waveCoord());
            auto matrixCoordLdsB = MappingLdsB::matrixCoord(MappingLdsB::waveCoord());

            auto* addrLdsA = MappingLdsA::dataCoord(baseAddrLdsA, ldLds, matrixCoordLdsA);
            auto* addrLdsB = MappingLdsA::dataCoord(baseAddrLdsB, ldLds, matrixCoordLdsB);

            wmma::store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragA), ldLds);
            wmma::store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragB), ldLds);

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
                wmma::synchronize_workgroup();
                wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
                wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);

                // Start pulling in the next block
                auto fragANext = FragA();
                auto fragBNext = FragB();
                wmma::load_matrix_sync(fragANext, addrA, lda);
                wmma::load_matrix_sync(fragBNext, addrB, ldb);

                // Mma for current block
                wmma::synchronize_workgroup();
                wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

                wmma::store_matrix_sync(addrLdsA, reinterpret_cast<FragLdsA&>(fragANext), ldLds);
                wmma::store_matrix_sync(addrLdsB, reinterpret_cast<FragLdsB&>(fragBNext), ldLds);

                addrA += incrA;
                addrB += incrB;
            }

            // Mma for the last block
            wmma::synchronize_workgroup();
            wmma::load_matrix_sync(reinterpret_cast<FragLdsA&>(fragA), addrLdsA, ldLds);
            wmma::load_matrix_sync(reinterpret_cast<FragLdsB&>(fragB), addrLdsB, ldLds);
            wmma::synchronize_workgroup();
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

template<uint32_t BlockM, uint32_t BlockN, uint32_t BlockK,
         typename InputT, typename OutputT, typename ComputeT,
         typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
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
        // Minimum matrix sizes
        if(m < BlockM * TBlockX / AMDGCN_WAVE_SIZE || n < BlockN * TBlockY || k < BlockK)
        {
            return;
        }

        // Max LDS usage
        if(LDS_MAX_BYTES
        < sizeof(InputT) * (blockDim.x / 64 * blockDim.y) * (BlockN * BlockK + BlockM * BlockK))
        {
            return;
        }

        TBlockX = TBlockXI;
        TBlockY = TBlockYI;
        M = m;
        N = n;
        K = k;
        Alpha = alpha;
        Beta = beta;

        int lda = std::is_same<LayoutA, row_major>::value ? k : m;
        int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
        int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
        int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

        std::cout << TBlockX << ", " << TBlockY << ", " << BlockM << ", " << BlockN << ", " << BlockK
              << ", " << m << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
              << ", " << beta << ", " << ldc << ", " << ldd << ", "
              << (std::is_same<LayoutA, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutB, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutC, row_major>::value ? "R" : "C") << "_"
              << (std::is_same<LayoutD, row_major>::value ? "R" : "C") << ", "
              << dataTypeToString<InputT>() << "_" << dataTypeToString<OutputT>() << "_"
              << dataTypeToString<ComputeT>() <<std::endl;

        // Initialize input matrices
        matrixA.resize(M * K);
        matrixB.resize(K * N);
        matrixC.resize(M * N);
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

        auto gridDim
            = dim3(ceilDiv(M, BlockM * TBlockX / AMDGCN_WAVE_SIZE), ceilDiv(N, BlockN * TBlockY));

        auto blockDim = dim3(TBlockX, TBlockY);
     }

    void barrierTestWrapper()
    {
        // Minimum matrix sizes
        if(M < BlockM * TBlockX / AMDGCN_WAVE_SIZE || N < BlockN * TBlockY || K < BlockK)
        {
            return;
        }

        // Max LDS usage
        if(LDS_MAX_BYTES
        < sizeof(InputT) * (blockDim.x / 64 * blockDim.y) * (BlockN * BlockK + BlockM * BlockK))
        {
            return;
        }

        hipLaunchKernelGGL((barrierTest<BlockM,
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
                              sizeof(InputT) * (blockDim.x / 64 * blockDim.y)
                              * (BlockN * BlockK + BlockM * BlockK), // sharedMemBytes
                              0, // stream
                              M,
                              N,
                              K,
                              d_a,
                              d_b,
                              d_c,
                              d_d,
                              Alpha,
                              Beta);
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
        < sizeof(InputT) * (blockDim.x / 64 * blockDim.y) * (BlockN * BlockK + BlockM * BlockK))
        {
            return;
        }

        // Init reference data and then validate
        matrixD_ref.resize(M * N);
        const size_t bytesD = matrixD_ref.size() * sizeof(OutputT);

        CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));
    
        double errorTolerance = sizeof(ComputeT) < sizeof(float32_t) ? 100.0 : 10.0;
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

        // Release device memory
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
    }

    private:
        uint32_t M, N, K, TBlockX, TBlockY, Alpha, Beta;
        InputT *d_a, *d_b;
        OutputT *d_c, *d_d;

        dim3 gridDim, blockDim;
        std::vector<InputT> matrixA, matrixB;
        std::vector<OutputT> matrixC, matrixD, matrixD_ref;
};

template <typename T>
struct BarrierTestWrapper;

template <typename BlockM,
          typename BlockN,
          typename BlockK,
          typename InputT,
          typename LayoutA,
          typename LayoutB>
struct BarrierTestWrapper<std::tuple<BlockM, BlockN, BlockK, InputT, LayoutA, LayoutB>> : public testing::Test
{
    Kernel<BlockM::value, BlockN::value, BlockK::value, InputT, InputT, InputT,
            LayoutA, LayoutB, LayoutA, LayoutA> *obj[NO_OF_TESTS];
        
    void SetUp() override 
    {
    std::vector<std::array<int, 2>> thread_block = {{64, 1}, {64, 2}, {64, 4},
                                                    {128,1}, {128,2},
                                                    {256,1}};

        // For fills, we must have the same geometry for all matrices
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
                obj[i++] = new Kernel<BlockM::value, BlockN::value, BlockK::value, 
                                      InputT, InputT, InputT, LayoutA, LayoutB, LayoutA, LayoutA>
                            (tblock[0], tblock[1], size[0], size[1], size[2], 0, 0);
            }
        }
    }

    void barrierTestSetup()
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
            if(obj[i] != NULL)
                obj[i]->barrierTestWrapper();
    }

    void TearDown() override
    {
        for(int i = 0; i < NO_OF_TESTS; i++)
        {
            if(obj[i] != NULL)
            {
                delete obj[i];
                obj[i] = NULL;
            }
        }
    }
};

using Implementations = testing::Types<
    // BlockM, BlockN, BlockK, InputT/OutputT/ComputeT, LayoutA/LayoutC/LayoutD, LayoutB/*
    std::tuple<I<16>, I<16>, I<16>, float32_t, col_major, row_major>,
    std::tuple<I<16>, I<16>, I<16>, float16_t, col_major, row_major>,
    std::tuple<I<16>, I<16>, I<16>, hfloat16_t, col_major, row_major>,
    std::tuple<I<32>, I<32>, I<32>, float32_t, col_major, row_major>,
    std::tuple<I<32>, I<32>, I<32>, float16_t, col_major, row_major>,
    std::tuple<I<32>, I<32>, I<32>, hfloat16_t, col_major, row_major>>;

TYPED_TEST_SUITE(BarrierTestWrapper, Implementations);

TYPED_TEST(BarrierTestWrapper, barrier)
{
    this->barrierTestSetup();
}