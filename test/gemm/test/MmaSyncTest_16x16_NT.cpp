
#include "MmaSyncTest.h"

// Test params for 16 x 16 NT kernels
struct TestParams16x16NT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::col_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL + double
    // Block Sizes: 16 x 16 x BlockK
    // Layouts: NT
    using Types =
        typename Concat<typename Base::TestTypesIOC, typename Base::TestTypeDouble>::Result;
    using BlockSizes = typename Base::TestBlockSizes16x16;
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
class MmaSyncTest16x16NT : public MmaSyncTest
{
};

TEST_P(MmaSyncTest16x16NT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncTest16x16NT,
                         ::testing::Combine(::testing::ValuesIn(TestParams16x16NT::kernels()),
                                            ::testing::ValuesIn(TestParams16x16NT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams16x16NT::problemSizes()),
                                            ::testing::ValuesIn(TestParams16x16NT::alphas()),
                                            ::testing::ValuesIn(TestParams16x16NT::betas())));
