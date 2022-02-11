/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_DETAIL_STORE_CONTAMINATION_HPP
#define ROCWMMA_DETAIL_STORE_CONTAMINATION_HPP

#include "device/store_contamination.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct StoreContaminationKernel : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

    public:
        StoreContaminationKernel()          = default;
        virtual ~StoreContaminationKernel() = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probSize) final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Make some rectangular padding around the area of interest
            // Param1 = padM
            // Param2 = padN
            using SizeT  = typename Base::DataStorage::ProblemSize;
            using IndexT = typename std::tuple_element<0, SizeT>::type;
            auto paddedProbSize
                = std::make_pair(std::get<0>(probSize) + 2 * static_cast<IndexT>(Base::mParam1),
                                 std::get<1>(probSize) + 2 * static_cast<IndexT>(Base::mParam2));

            auto paddedElements = std::get<0>(paddedProbSize) * std::get<1>(paddedProbSize);

            // Initialize matrix storage with padded size.
            // Padded size >= MxN
            dataInstance->resizeStorage(paddedProbSize);

            // Initialize input data on host (not padded)
            MatrixUtil<Layout>::fill(dataInstance->hostIn().get(), Base::mM, Base::mN);
            dataInstance->copyData(
                dataInstance->deviceIn(), dataInstance->hostIn(), paddedElements);

            // Initialize output data on host (padded size, all with marker elements)
            auto outputInit = dataInstance->template allocHost<DataT>(paddedElements);
            MatrixUtil<Layout>::fill(outputInit.get(),
                                     std::get<0>(paddedProbSize),
                                     std::get<1>(paddedProbSize),
                                     std::numeric_limits<DataT>::max());
            dataInstance->copyData(dataInstance->deviceOut(), outputInit, paddedElements);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();
            auto& kernelResult = dataInstance->hostIn();

            // Re-use host in memory for result
            // Use padded MxN as output is padded
            auto          paddedSize     = dataInstance->problemSize();
            const int64_t paddedElements = std::get<0>(paddedSize) * std::get<1>(paddedSize);

            // Cache current kernel result from device
            dataInstance->copyData(kernelResult, dataInstance->deviceOut(), paddedElements);

            double errorTolerance = 1.0;

            // See if our output padding contains any contamination
            auto result = countPaddingVal<DataT, Layout>(kernelResult.get(),
                                                         Base::mM,
                                                         Base::mN,
                                                         static_cast<uint32_t>(Base::mParam1),
                                                         static_cast<uint32_t>(Base::mParam2),
                                                         std::numeric_limits<DataT>::max(),
                                                         errorTolerance);

            // All padding must be markers
            Base::mValidationResult = (result == (paddedElements - Base::mM * Base::mN));
        }

        typename Base::KernelFunc kernelImpl() const = 0;
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct StoreContaminationKernelA
        : public StoreContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = StoreContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(storeContaminationA<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct StoreContaminationKernelB
        : public StoreContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = StoreContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(storeContaminationB<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct StoreContaminationKernelAcc
        : public StoreContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = StoreContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(storeContaminationAcc<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <template <uint32_t, uint32_t, typename, typename> class KernelClass>
    struct StoreContaminationGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT  = 0,
            BlockM = 1,
            BlockN = 2,
            Layout = 3
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT = KernelClass<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                        std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                        std::tuple_element_t<DataT, TestParamsT>, // DataT
                                        std::tuple_element_t<Layout, TestParamsT> // Layout
                                        >;

            return std::make_shared<KernelT>();
        }
    };

    using StoreContaminationGeneratorA   = StoreContaminationGenerator<StoreContaminationKernelA>;
    using StoreContaminationGeneratorB   = StoreContaminationGenerator<StoreContaminationKernelB>;
    using StoreContaminationGeneratorAcc = StoreContaminationGenerator<StoreContaminationKernelAcc>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_STORE_CONTAMINATION_HPP
