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

#ifndef ROCWMMA_DETAIL_LOAD_CONTAMINATION_H
#define ROCWMMA_DETAIL_LOAD_CONTAMINATION_H

#include "UnitKernelBase.h"
#include "device/LoadContamination.h"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadContaminationKernel : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

    public:
        LoadContaminationKernel()          = default;
        virtual ~LoadContaminationKernel() = default;

    protected:
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

            // Initialize matrix storage with padded size.
            // Padded size >= MxN
            dataInstance->resizeStorage(paddedProbSize);

            // Initialize input data on host.
            // Initialize padding with contamination values
            MatrixUtil<Layout>::fill_with_padding(dataInstance->hostIn().get(),
                                                  Base::mM,
                                                  Base::mN,
                                                  Base::mParam1,
                                                  Base::mParam2,
                                                  std::numeric_limits<DataT>::max());

            // Padded MxN goes in for read, MxN result comes out
            dataInstance->copyData(dataInstance->deviceIn(),
                                   dataInstance->hostIn(),
                                   std::get<0>(paddedProbSize) * std::get<1>(paddedProbSize));
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Re-use host in memory for result
            // Use M x N as output is not padded
            const int64_t sizeD        = Base::mM * Base::mN;
            auto&         kernelResult = dataInstance->hostIn();

            // Cache current kernel result from device
            dataInstance->copyData(kernelResult, dataInstance->deviceOut(), sizeD);

            double errorTolerance = 1.0;

            // See if our output contains any contamination
            auto result = countVal(kernelResult.get(),
                                   Base::mM * Base::mN,
                                   std::numeric_limits<DataT>::max(),
                                   errorTolerance);

            // We want no contamination
            Base::mValidationResult = (result == 0);
        }

        virtual typename Base::KernelFunc kernelImpl() const = 0;
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadContaminationKernelA final
        : public LoadContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(loadContaminationA<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadContaminationKernelB final
        : public LoadContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(loadContaminationB<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadContaminationKernelAcc final
        : public LoadContaminationKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadContaminationKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(loadContaminationAcc<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <template <uint32_t, uint32_t, typename, typename> class KernelClass>
    struct LoadContaminationGenerator
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

    using LoadContaminationGeneratorA   = LoadContaminationGenerator<LoadContaminationKernelA>;
    using LoadContaminationGeneratorB   = LoadContaminationGenerator<LoadContaminationKernelB>;
    using LoadContaminationGeneratorAcc = LoadContaminationGenerator<LoadContaminationKernelAcc>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_LOAD_CONTAMINATION_H
