/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <tuple>
#include <type_traits>

#include "detail/unpackutil.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{

    template <typename GeneratorImpl>
    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        // Types: Base IOC + double
        using Types = std::tuple<
            int8_t, // use int8_t since float8_t will be skipped by default on some platform
            float16_t,
            float32_t,
            float64_t>;

        // Vector Width.
        using VWs = std::tuple<I<2>, I<4>, I<8>>;

        using KernelParams = typename CombineLists<VWs, Types>::Result;

        // Assemble the kernel generator
        // Kernel: VectorUtil
        // using GeneratorImpl   = UnpackLo2Generator;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            // clang-format off
            return { {warpSize, 1}, {warpSize * 2, 1}, {warpSize * 4, 1}};
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
            return { {1, 1} };
            // clang-format on
        }

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

    using UnpackLo2TestParams    = TestParams<UnpackLo2Generator>;
    using UnpackLo4TestParams    = TestParams<UnpackLo4Generator>;
    using UnpackLo8TestParams    = TestParams<UnpackLo8Generator>;
    using UnpackHi2TestParams    = TestParams<UnpackHi2Generator>;
    using UnpackHi4TestParams    = TestParams<UnpackHi4Generator>;
    using UnpackHi8TestParams    = TestParams<UnpackHi8Generator>;
    using UnpackLoHi2TestParams  = TestParams<UnpackLoHi2Generator>;
    using UnpackLoHi4TestParams  = TestParams<UnpackLoHi4Generator>;
    using UnpackLoHi8TestParams  = TestParams<UnpackLoHi8Generator>;
    using UnpackLoHi16TestParams = TestParams<UnpackLoHi16Generator>;
    using UnpackLoHi32TestParams = TestParams<UnpackLoHi32Generator>;
} // namespace rocwmma

// Test suite for unpackLo2
class UnpackLo2Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLo2Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLo2Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLo2TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLo2TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLo2TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLo2TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLo2TestParams::param2s())));

// Test suite for unpackLo4
class UnpackLo4Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLo4Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLo4Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLo4TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLo4TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLo4TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLo4TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLo4TestParams::param2s())));

// Test suite for unpackLo8
class UnpackLo8Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLo8Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLo8Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLo8TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLo8TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLo8TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLo8TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLo8TestParams::param2s())));

// Test suite for unpackHi2
class UnpackHi2Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackHi2Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackHi2Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackHi2TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackHi2TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackHi2TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackHi2TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackHi2TestParams::param2s())));

// Test suite for unpackHi4
class UnpackHi4Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackHi4Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackHi4Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackHi4TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackHi4TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackHi4TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackHi4TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackHi4TestParams::param2s())));

// Test suite for unpackHi8
class UnpackHi8Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackHi8Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackHi8Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackHi8TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackHi8TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackHi8TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackHi8TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackHi8TestParams::param2s())));

// Test suite for unpackLoHi2
class UnpackLoHi2Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLoHi2Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLoHi2Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLoHi2TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi2TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi2TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi2TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi2TestParams::param2s())));

// Test suite for unpackLoHi4
class UnpackLoHi4Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLoHi4Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLoHi4Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLoHi4TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi4TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi4TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi4TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi4TestParams::param2s())));

// Test suite for unpackLoHi8
class UnpackLoHi8Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLoHi8Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLoHi8Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLoHi8TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi8TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi8TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi8TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi8TestParams::param2s())));

// Test suite for unpackLoHi16
class UnpackLoHi16Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLoHi16Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLoHi16Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLoHi16TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi16TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi16TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi16TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi16TestParams::param2s())));

// Test suite for unpackLoHi32
class UnpackLoHi32Test : public rocwmma::UnitTest
{
};

TEST_P(UnpackLoHi32Test, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    KernelTests,
    UnpackLoHi32Test,
    ::testing::Combine(::testing::ValuesIn(rocwmma::UnpackLoHi32TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi32TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi32TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi32TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::UnpackLoHi32TestParams::param2s())));
