/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#ifndef WMMA_DETAIL_MAP_THREAD_TO_MATRIX_H
#define WMMA_DETAIL_MAP_THREAD_TO_MATRIX_H

#include "UnitKernelBase.h"
#include "device/MapThreadToMatrix.h"

// Wrapper into the actual device function
template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
struct MapThreadToMatrixKernel final : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
{
private:
    using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

public:
    MapThreadToMatrixKernel() {}
    ~MapThreadToMatrixKernel() final {}

    void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
    {
        auto& dataInstance = Base::DataStorage::instance();

        // Initialize matrix storage
        const int64_t sizeD = Base::mM * Base::mN;
        dataInstance->resizeStorage(probsize);

        // Initialize matrix data on host
        MatrixUtil<Layout>::fill(dataInstance->hostIn().get(), Base::mM, Base::mN);

        dataInstance->copyData(dataInstance->deviceIn(), dataInstance->hostIn(), sizeD);
    }

    void validateResultsImpl() final
    {
        auto& dataInstance = Base::DataStorage::instance();

        // Allocated managed memory for results on host
        const int64_t sizeD = Base::mM * Base::mN;

        auto kernelResult = dataInstance->allocHost(sizeD);

        // Cache current kernel result from device
        dataInstance->copyData(kernelResult, dataInstance->deviceOut(), sizeD);

        double errorTolerance = 10.0;

        std::tie(Base::mValidationResult, Base::mMaxRelativeError)
            = compareEqual<DataT, DataT, Layout, Layout>(kernelResult.get(),
                                                         dataInstance->hostIn().get(),
                                                         Base::mM,
                                                         Base::mN,
                                                         errorTolerance);

        EXPECT_TRUE(Base::mValidationResult) << "Max relative error: " << Base::mMaxRelativeError;
    }

    typename Base::KernelFunc kernelImpl() const final
    {
        return typename Base::KernelFunc(MapThreadToMatrix<BlockM, BlockN, DataT, Layout>);
    }
};

// This is the GeneratorImpl class
struct MapThreadToMatrixGenerator
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
        using KernelT
            = MapThreadToMatrixKernel<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                      std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                      std::tuple_element_t<DataT, TestParamsT>, // DataT
                                      std::tuple_element_t<Layout, TestParamsT> // Layout
                                      >;

        return std::make_shared<KernelT>();
    }
};

#endif // WMMA_DETAIL_MAP_THREAD_TO_MATRIX_H
