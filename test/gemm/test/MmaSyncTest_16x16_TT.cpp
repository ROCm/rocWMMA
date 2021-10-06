
#include "MmaSyncTest.h"

// Test params for 16 x 16 TT kernels
struct TestParams16x16TT : public CommonTestParams
{
    using ABLayouts = std::tuple<wmma::row_major, wmma::row_major>;
    using Base      = CommonTestParams;

    // Set up the testing context:
    // Kernel: MmaSync
    // Types: ALL + double
    // Block Sizes: 16 x 16 x BlockK
    // Layouts: TT
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
class MmaSyncTest16x16TT : public MmaSyncTest
{
};

TEST_P(MmaSyncTest16x16TT, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(GemmKernelTests,
                         MmaSyncTest16x16TT,
                         ::testing::Combine(::testing::ValuesIn(TestParams16x16TT::kernels()),
                                            ::testing::ValuesIn(TestParams16x16TT::threadBlocks()),
                                            ::testing::ValuesIn(TestParams16x16TT::problemSizes()),
                                            ::testing::ValuesIn(TestParams16x16TT::alphas()),
                                            ::testing::ValuesIn(TestParams16x16TT::betas())));
