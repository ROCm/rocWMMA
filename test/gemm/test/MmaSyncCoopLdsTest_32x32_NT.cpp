
#include "MmaSyncCoopLdsTest.h"

// Test params for 32 x 32 NT kernels
struct TestParams32x32NT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::col_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL - double
    // Block Sizes: 32 x 32 x BlockK
    // Layouts: NT
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
class MmaSyncCoopLdsTest32x32NT : public MmaSyncCoopLdsTest
{
};

TEST_P(MmaSyncCoopLdsTest32x32NT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncCoopLdsTest32x32NT,
                         ::testing::Combine(::testing::ValuesIn(TestParams32x32NT::kernels()),
                                            ::testing::ValuesIn(TestParams32x32NT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams32x32NT::problemSizes()),
                                            ::testing::ValuesIn(TestParams32x32NT::alphas()),
                                            ::testing::ValuesIn(TestParams32x32NT::betas())));
