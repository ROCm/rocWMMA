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

#ifndef ROCWMMA_DETAIL_MAP_BLOCK_TO_MATRIX_OVERRIDE_N_HPP
#define ROCWMMA_DETAIL_MAP_BLOCK_TO_MATRIX_OVERRIDE_N_HPP

#include "device/map_block_to_matrix_override_n.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct MapBlockToMatrixOverrideNKernel final
        : public UnitKernelBase<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, Layout>;

    public:
        MapBlockToMatrixOverrideNKernel()        = default;
        ~MapBlockToMatrixOverrideNKernel() final = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            auto& dataInstance = Base::DataStorage::instance();

            srand((unsigned)time(0));
            uint32_t nBlocks = this->gridDim().y;
            Base::mParam1    = static_cast<DataT>(static_cast<float32_t>(rand() % nBlocks));

            // Initialize matrix storage
            const int64_t sizeD = Base::mM * Base::mN;
            dataInstance->resizeStorage(probsize);

            // Initialize matrix data on host
            MatrixUtil<Layout>::template fill<DataT>(
                dataInstance->hostIn().get(), Base::mM, Base::mN, (DataT)0);
            dataInstance->copyData(dataInstance->deviceOut(), dataInstance->hostIn(), sizeD);

            MatrixUtil<Layout>::fill(dataInstance->hostIn().get(), Base::mM, Base::mN);
            dataInstance->copyData(dataInstance->deviceIn(), dataInstance->hostIn(), sizeD);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Allocated managed memory for results on host
            const int64_t sizeD = Base::mM * Base::mN;

            auto kernelResult = dataInstance->template allocHost<DataT>(sizeD);
            auto hostResult   = dataInstance->hostIn().get();

            //Allocate additional resource to validate only the overriden col of size Base::mM
            auto kernelResultToValidate = dataInstance->template allocHost<DataT>(Base::mM);
            auto hostResultToValidate   = dataInstance->template allocHost<DataT>(Base::mM);

            // Cache current kernel result from device
            dataInstance->copyData(kernelResult, dataInstance->deviceOut(), sizeD);

            // Cache current kernel result from device
            dataInstance->copyData(kernelResult, dataInstance->deviceOut(), sizeD);

            double   errorTolerance = 1.0;
            uint32_t baseOffset
                = std::is_same<Layout, row_major>::value
                      ? static_cast<uint32_t>(static_cast<float32_t>(Base::mParam1)) * BlockN
                      : static_cast<uint32_t>(static_cast<float32_t>(Base::mParam1)) * Base::mM
                            * BlockN;
            uint32_t ld = std::is_same<Layout, row_major>::value ? Base::mN : 1;

            // Validation offset starts at col Base::mParam1 * BlockN
            // To get to col Base::mParam1 * BlockN,
            // in case of row-major layout, skip only Base::mParam1 * BlockN elements
            // in case of col-major layout, skip Base::mParam1 * Base::mM * BlockN elements
            uint32_t baseOffset
                = std::is_same<Layout, row_major>::value
                      ? static_cast<uint32_t>(static_cast<float32_t>(Base::mParam1)) * BlockN
                      : static_cast<uint32_t>(static_cast<float32_t>(Base::mParam1)) * Base::mM
                            * BlockN;

            // To get to the elements across col,
            // in case of row-major, next element access is by current_element + Base::mN
            // in case of col-major, next element access is current_element + 1
            uint32_t ld = std::is_same<Layout, row_major>::value ? Base::mN : 1;

            // Copy the entire col Base::mParam1 * BlockN
            for(int i = 0; i < Base::mM; i++)
            {
                kernelResultToValidate[i] = kernelResult[baseOffset + (i * ld)];
                hostResultToValidate[i]   = hostResult[baseOffset + (i * ld)];
            }

            std::tie(Base::mValidationResult, Base::mMaxRelativeError)
                = compareEqual<DataT, DataT, Layout, Layout>(kernelResultToValidate.get(),
                                                             hostResultToValidate.get(),
                                                             Base::mM,
                                                             1,
                                                             errorTolerance);
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return
                typename Base::KernelFunc(MapBlockToMatrixOverrideN<BlockM, BlockN, DataT, Layout>);
        }
    };

    // This is the GeneratorImpl class
    struct MapBlockToMatrixOverrideNGenerator
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
            using KernelT     = MapBlockToMatrixOverrideNKernel<
                std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                std::tuple_element_t<DataT, TestParamsT>, // DataT
                std::tuple_element_t<Layout, TestParamsT> // Layout
                >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_MAP_BLOCK_TO_MATRIX_OVERRIDE_N_HPP
