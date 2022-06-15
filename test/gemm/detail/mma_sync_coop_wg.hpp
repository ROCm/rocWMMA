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

#ifndef ROCWMMA_DETAIL_MMA_SYNC_COOP_WG_HPP
#define ROCWMMA_DETAIL_MMA_SYNC_COOP_WG_HPP

#include "device/mma_sync_multi_lds.hpp"
#include "gemm_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD,
              typename LayoutLds,
              typename GemmConfig,
              uint32_t BlocksX = 1,
              uint32_t BlocksY = 1>
    struct MmaSyncCoopWgKernel final : public GemmKernelBase<BlockM,
                                                             BlockN,
                                                             BlockK,
                                                             InputT,
                                                             OutputT,
                                                             ComputeT,
                                                             LayoutA,
                                                             LayoutB,
                                                             LayoutC,
                                                             LayoutD>
    {
    private:
        using Base = GemmKernelBase<BlockM,
                                    BlockN,
                                    BlockK,
                                    InputT,
                                    OutputT,
                                    ComputeT,
                                    LayoutA,
                                    LayoutB,
                                    LayoutC,
                                    LayoutD>;

    public:
        MmaSyncCoopWgKernel() {}
        ~MmaSyncCoopWgKernel() final {}

        dim3 gridDim() const final
        {
            return dim3(ceilDiv(Base::mM, BlockM * BlocksX * Base::mTBlockX / AMDGCN_WAVE_SIZE),
                        ceilDiv(Base::mN, BlockN * BlocksY * Base::mTBlockY));
        }

        bool checkSizes() const final
        {
            return ((BlockM * BlocksX * Base::mTBlockX / AMDGCN_WAVE_SIZE) <= Base::mM)
                   && ((BlockN * BlocksY * Base::mTBlockY) <= Base::mN) && (BlockK <= Base::mK);
        }

        bool checkQuirks() const final
        {
            // Don't run the kernel if the threadblock size is not supported
            return (kernelImpl() != nullptr);
        }

        // Lds memory usage in bytes
        uint32_t ldsUsage() const final
        {
            // Uses 2 lds blocks for prefetch loop
            return 2 * sizeof(InputT)
                   * (Base::mTBlockX / AMDGCN_WAVE_SIZE * BlocksX * BlockM * BlockK
                      + Base::mTBlockY * BlocksY * BlockK * BlockN);
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            auto kernelFunc = typename Base::KernelFunc(nullptr);

            if(Base::mTBlockX == 64)
            {
                if(Base::mTBlockY == 1)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           64,
                                                                           1>);
                }
                else if(Base::mTBlockY == 2)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           64,
                                                                           2>);
                }
                else if(Base::mTBlockY == 4)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           64,
                                                                           4>);
                }
            }
            else if(Base::mTBlockX == 128)
            {
                if(Base::mTBlockY == 1)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           128,
                                                                           1>);
                }
                else if(Base::mTBlockY == 2)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           128,
                                                                           2>);
                }
            }
            else if(Base::mTBlockX == 256)
            {
                if(Base::mTBlockY == 1)
                {
                    kernelFunc = typename Base::KernelFunc(mmaSyncMultiLds<BlockM,
                                                                           BlockN,
                                                                           BlockK,
                                                                           InputT,
                                                                           OutputT,
                                                                           ComputeT,
                                                                           LayoutA,
                                                                           LayoutB,
                                                                           LayoutC,
                                                                           LayoutD,
                                                                           LayoutLds,
                                                                           GemmConfig,
                                                                           BlocksX,
                                                                           BlocksY,
                                                                           256,
                                                                           1>);
                }
            }

            return kernelFunc;
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const final
        {
            return Base::printHeader(stream << "GemmConfig, LytLds, BlocksX, BlocksY, ");
        }

        std::ostream& printKernel(std::ostream& stream = std::cout) const final
        {
            return Base::printKernel(stream << dataTypeToString<GemmConfig>() << ", "
                                            << dataTypeToString<LayoutLds>() << ", " << BlocksX
                                            << ", " << BlocksY << ", ");
        }
    };

    // This is the GeneratorImpl class for MmaSyncCoopLds
    struct MmaSyncCoopWgGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            InputT     = 0,
            OutputT    = 1,
            ComputeT   = 2,
            BlockM     = 3,
            BlockN     = 4,
            BlockK     = 5,
            LayoutA    = 6,
            LayoutB    = 7,
            LayoutCD   = 8,
            LayoutLds  = 9,
            GemmConfig = 10,
            BlocksX    = 11,
            BlocksY    = 12
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = MmaSyncCoopWgKernel<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                                      std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                                      std::tuple_element_t<BlockK, TestParamsT>::value, // BlockK
                                      std::tuple_element_t<InputT, TestParamsT>, // InputT
                                      std::tuple_element_t<OutputT, TestParamsT>, // OutputT
                                      std::tuple_element_t<ComputeT, TestParamsT>, // ComputeT
                                      std::tuple_element_t<LayoutA, TestParamsT>, // LayoutA
                                      std::tuple_element_t<LayoutB, TestParamsT>, // LayoutB
                                      std::tuple_element_t<LayoutCD, TestParamsT>, // LayoutC
                                      std::tuple_element_t<LayoutCD, TestParamsT>, // LayoutD
                                      std::tuple_element_t<LayoutLds, TestParamsT>, // LayoutLds
                                      std::tuple_element_t<GemmConfig, TestParamsT>, // LayoutLds
                                      std::tuple_element_t<BlocksX, TestParamsT>::value, // BlocksX
                                      std::tuple_element_t<BlocksY, TestParamsT>::value // BlocksY
                                      >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_MMA_SYNC_COOP_WG_HPP
