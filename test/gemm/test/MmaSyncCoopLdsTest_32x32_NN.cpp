
#include "MmaSyncCoopLdsTest.h"

// Test params for 32 x 32 NN kernels
struct TestParams32x32NN : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::col_major, wmma::col_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL - double
    // Block Sizes: 32 x 32 x BlockK
    // Layouts: NN
    using Types      = typename Base::TestTypesIOC;
    using BlockSizes = typename Base::TestBlockSizes32x32;
    using Layouts    = typename CombineOne<ABLayouts, typename Base::TestLayoutTypes>::Result;

    // Assemble the kernel generator
    using TestParams =
        typename CombineMany<Types, typename CombineMany<BlockSizes, Layouts>::Result>::Result;
    using GeneratorImpl   = MmaSyncCoopLdsGenerator;
    using KernelGenerator = KernelGenerator<TestParams, GeneratorImpl>;

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class MmaSyncCoopLdsTest32x32NN : public MmaSyncCoopLdsTest
{
};

TEST_P(MmaSyncCoopLdsTest32x32NN, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncCoopLdsTest32x32NN,
                         ::testing::Combine(::testing::ValuesIn(TestParams32x32NN::kernels()),
                                            ::testing::ValuesIn(TestParams32x32NN::threadBlocks()),
                                            ::testing::ValuesIn(TestParams32x32NN::problemSizes()),
                                            ::testing::ValuesIn(TestParams32x32NN::alphas()),
                                            ::testing::ValuesIn(TestParams32x32NN::betas())));
