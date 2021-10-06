
#include "MmaSyncLdsTest.h"

// Test params for 32 x 32 TN kernels
struct TestParams32x32TN : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::row_major, wmma::col_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL - double
    // Block Sizes: 32 x 32 x BlockK
    // Layouts: TN
    using Types      = typename Base::TestTypesIOC;
    using BlockSizes = typename Base::TestBlockSizes32x32;
    using Layouts    = typename CombineOne<ABLayouts, typename Base::TestLayoutTypes>::Result;

    // Assemble the kernel generator
    using TestParams =
        typename CombineMany<Types, typename CombineMany<BlockSizes, Layouts>::Result>::Result;
    using GeneratorImpl   = MmaSyncLdsGenerator;
    using KernelGenerator = KernelGenerator<TestParams, GeneratorImpl>;

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class MmaSyncLdsTest32x32TN : public MmaSyncLdsTest
{
};

TEST_P(MmaSyncLdsTest32x32TN, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncLdsTest32x32TN,
                         ::testing::Combine(::testing::ValuesIn(TestParams32x32TN::kernels()),
                                            ::testing::ValuesIn(TestParams32x32TN::threadBlocks()),
                                            ::testing::ValuesIn(TestParams32x32TN::problemSizes()),
                                            ::testing::ValuesIn(TestParams32x32TN::alphas()),
                                            ::testing::ValuesIn(TestParams32x32TN::betas())));
