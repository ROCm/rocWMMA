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

#ifndef ROCWMMA_DETAIL_LOAD_STORE_MATRIX_SYNC_HPP
#define ROCWMMA_DETAIL_LOAD_STORE_MATRIX_SYNC_HPP

#include "device/load_store_matrix_sync.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixSyncKernel : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

    public:
        LoadStoreMatrixSyncKernel()          = default;
        virtual ~LoadStoreMatrixSyncKernel() = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Initialize matrix storage
            const int64_t sizeD = Base::mM * Base::mN;
            dataInstance->resizeStorage(probsize);

            // Initialize data on host
            MatrixUtil<Layout>::fill(dataInstance->hostIn().get(), Base::mM, Base::mN);
            MatrixUtil<Layout>::fill(
                dataInstance->hostOut().get(), Base::mM, Base::mN, std::numeric_limits<DataT>::signaling_NaN());

            // Copy init data to device
            dataInstance->copyData(dataInstance->deviceIn(), dataInstance->hostIn(), sizeD);
            dataInstance->copyData(dataInstance->deviceOut(), dataInstance->hostOut(), sizeD);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            const int64_t sizeD = Base::mM * Base::mN;

            // Cache current kernel result from device
            dataInstance->copyData(dataInstance->hostOut(), dataInstance->deviceOut(), sizeD);

            double errorTolerance = 10.0;

            std::tie(Base::mValidationResult, Base::mMaxRelativeError)
                = compareEqual<DataT, DataT, Layout, Layout>(dataInstance->hostOut().get(),
                                                             dataInstance->hostIn().get(),
                                                             Base::mM,
                                                             Base::mN,
                                                             errorTolerance);
        }

        virtual typename Base::KernelFunc kernelImpl() const = 0;
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixSyncKernelA final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(LoadStoreMatrixSyncA<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixSyncKernelB final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(LoadStoreMatrixSyncB<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixSyncKernelAcc final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(LoadStoreMatrixSyncAcc<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <template <uint32_t, uint32_t, typename, typename> class KernelClass>
    struct LoadStoreMatrixSyncGenerator
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

    using LoadStoreMatrixSyncGeneratorA = LoadStoreMatrixSyncGenerator<LoadStoreMatrixSyncKernelA>;
    using LoadStoreMatrixSyncGeneratorB = LoadStoreMatrixSyncGenerator<LoadStoreMatrixSyncKernelB>;
    using LoadStoreMatrixSyncGeneratorAcc
        = LoadStoreMatrixSyncGenerator<LoadStoreMatrixSyncKernelAcc>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_LOAD_STORE_MATRIX_SYNC_HPP
