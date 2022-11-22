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

#ifndef ROCWMMA_DETAIL_IO_TRAITS_HPP
#define ROCWMMA_DETAIL_IO_TRAITS_HPP

#include "unit_kernel_base.hpp"

namespace rocwmma
{
    static constexpr uint32_t ERROR_VALUE   = 7;
    static constexpr uint32_t SUCCESS_VALUE = 0;

    // Wrapper into the actual device function
    template <typename DataT, uint32_t BlockDim, uint32_t BlockK, uint32_t VectorWidth>
    struct IOTraitsKernel final : public UnitKernelBase<BlockDim, BlockK, DataT, col_major>
    {
    private:
        using layout = col_major; // unused layout
        using Base   = UnitKernelBase<BlockDim, BlockK, DataT, col_major>;

    public:
        IOTraitsKernel()        = default;
        ~IOTraitsKernel() final = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            dataInstance->hostOut().get()[0] = static_cast<DataT>(ERROR_VALUE);
        }

        void exec() final
        {
            if(Base::mRunFlag)
            {
                bool err = false;

                using PackTraits = detail::PackTraits<DataT>;

                // Check on pack ratio sizes
                err |= (PackTraits::PackRatio * sizeof(typename PackTraits::UnpackedT)
                        != sizeof(typename PackTraits::PackedT));
                err |= (!std::is_same<DataT, typename PackTraits::UnpackedT>::value);

                // Check consistency of packed vs unpacked types
                if(std::is_floating_point<typename PackTraits::UnpackedT>::value)
                {
                    err |= (!std::is_floating_point<typename PackTraits::PackedT>::value);
                }
                else if(std::is_integral<typename PackTraits::UnpackedT>::value)
                {
                    err |= (!std::is_integral<typename PackTraits::PackedT>::value);
                }

                // VectorWidthTraits, C++ perspective
                using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
                err |= (IOTraits::ThreadsPerIO != AMDGCN_WAVE_SIZE);
                err |= (IOTraits::ElementsPerIO != (AMDGCN_WAVE_SIZE * VectorWidth));
                err |= (IOTraits::KPerIO
                        != std::max(1u, (AMDGCN_WAVE_SIZE * VectorWidth / BlockDim)));
                err |= (IOTraits::ElementCount != (BlockDim * BlockK));
                err |= (IOTraits::IOCount
                        != (BlockDim * BlockK / (AMDGCN_WAVE_SIZE * VectorWidth)));
                err |= (IOTraits::UnpackedSize != (BlockDim * BlockK / AMDGCN_WAVE_SIZE));
                err |= (IOTraits::PackedSize
                        != (BlockDim * BlockK / AMDGCN_WAVE_SIZE / PackTraits::PackRatio));

                // Physical hardware perspective
                err |= (IOTraits::UnpackedVRegCount
                        != (IOTraits::UnpackedSize
                            * std::max(1u, (uint32_t)sizeof(DataT) / AMDGCN_DWORD_SIZE_BYTES)));
                err |= (IOTraits::PackedVRegCount
                        != (IOTraits::PackedSize
                            * std::max(1u, (uint32_t)sizeof(DataT) / AMDGCN_DWORD_SIZE_BYTES)));

                if(!err)
                {
                    auto& dataInstance               = Base::DataStorage::instance();
                    dataInstance->hostOut().get()[0] = static_cast<DataT>(SUCCESS_VALUE);
                }
            }
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Check the single output result
            Base::mValidationResult = (dataInstance->hostOut().get()[0] == DataT(SUCCESS_VALUE));
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(nullptr);
        }
    };

    // This is the GeneratorImpl class
    struct IOTraitsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            BlockM      = 1,
            BlockN      = 2,
            VectorWidth = 3
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = IOTraitsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                 std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                 std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                 std::tuple_element_t<VectorWidth, TestParamsT>::value // Layout
                                 >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_IO_TRAITS_HPP
