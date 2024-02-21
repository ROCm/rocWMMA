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

#ifndef ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP
#define ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP

#include "device/transforms.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t K, uint32_t VW, typename DataT>
    struct TransformsKernel
        : public UnitKernelBase<1,
                                1,
                                DataT,
                                col_major> // BlockM, BlockN, DataLayout are redundant for this test
    {
    private:
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        TransformsKernel()  = default;
        ~TransformsKernel() = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            dataInstance->hostOut().get()[0] = static_cast<DataT>(ERROR_VALUE);
            dataInstance->copyData(dataInstance->deviceOut(), dataInstance->hostOut(), 1);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Cache current kernel result from device
            dataInstance->copyData(dataInstance->hostOut(), dataInstance->deviceOut(), 1);

            // Check the single output result
            Base::mValidationResult = (dataInstance->hostOut().get()[0] == DataT(SUCCESS_VALUE));
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const override
        {
            return stream << "WSize, DataT, VW, K" << std::endl;
        }
        std::ostream& printKernel(std::ostream& stream = std::cout) const override
        {
            using DeviceInfo = HipDevice;
            stream << "w" << DeviceInfo::instance()->warpSize() << ", " << dataTypeToString<DataT>()
                   << ", " << VW << ", " << K;

            return stream;
        }
    };

    template <uint32_t K, uint32_t VW, typename DataT>
    struct AossoaKernel final : public TransformsKernel<K, VW, DataT>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;
        typename Base::KernelFunc kernelImpl() const override final
        {
            return typename Base::KernelFunc(aossoaTest<DataT, VW, K>);
        }
    };

    template <uint32_t K, uint32_t VW, typename DataT>
    struct SoaaosKernel final : public TransformsKernel<K, VW, DataT>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;
        typename Base::KernelFunc kernelImpl() const override final
        {
            return typename Base::KernelFunc(soaaosTest<DataT, VW, K>);
        }
    };

    // This is the GeneratorImpl class
    template <template <uint32_t K, uint32_t VW, typename DataT> typename Func>
    struct TransformsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            K     = 0,
            VW    = 1,
            DataT = 2,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = Func<std::tuple_element_t<K, TestParamsT>::value, // K
                                 std::tuple_element_t<VW, TestParamsT>::value, // VW
                                 std::tuple_element_t<DataT, TestParamsT> // DataT
                                 >;

            return std::make_shared<KernelT>();
        }
    };
    using AossoaGenerator = TransformsGenerator<AossoaKernel>;
    using SoaaosGenerator = TransformsGenerator<SoaaosKernel>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_TRANSFORMS_TEST_HPP
