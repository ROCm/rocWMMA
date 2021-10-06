
#include "MmaSyncTest.h"

// Test params for 32 x 32 TT kernels
struct TestParams32x32TT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::row_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL - double
    // Block Sizes: 32 x 32 x BlockK
    // Layouts: TT
    using Types      = typename Base::TestTypesIOC;
    using BlockSizes = typename Base::TestBlockSizes32x32;
    using Layouts    = typename CombineOne<ABLayouts, typename Base::TestLayoutTypes>::Result;

    // Assemble the kernel generator
    using TestParams =
        typename CombineMany<Types, typename CombineMany<BlockSizes, Layouts>::Result>::Result;
    using GeneratorImpl   = MmaSyncGenerator;
    using KernelGenerator = KernelGenerator<TestParams, GeneratorImpl>;

    static inline typename KernelGenerator::ResultT kernels()
    {
        return KernelGenerator::generate();
    }
};

// Test suite for unique parameterization
class MmaSyncTest32x32TT : public MmaSyncTest
{
};

TEST_P(MmaSyncTest32x32TT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncTest32x32TT,
                         ::testing::Combine(::testing::ValuesIn(TestParams32x32TT::kernels()),
                                            ::testing::ValuesIn(TestParams32x32TT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams32x32TT::problemSizes()),
                                            ::testing::ValuesIn(TestParams32x32TT::alphas()),
                                            ::testing::ValuesIn(TestParams32x32TT::betas())));
