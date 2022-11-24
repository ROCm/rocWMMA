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

#ifndef ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP
#define ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP

#include "device/cross_lane_ops.hpp"
#include "reference.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask  = 0xF,
              uint32_t WriteBankMask = 0xF,
              bool     BoundCtrl     = false>
    struct CrossLaneOpsKernelBase
        : public UnitKernelBase<1,
                                1,
                                DataT,
                                col_major> // BlockM, BlockN, DataLayout are redundant for this test
    {
    protected:
        using Base   = UnitKernelBase<1, 1, DataT, col_major>;
        using Layout = col_major;

    public:
        CrossLaneOpsKernelBase()          = default;
        virtual ~CrossLaneOpsKernelBase() = default;

        dim3 gridDim() const final
        {
            // Need to address the input array as 32b elements per thread.
            auto x
                = dim3(static_cast<uint32_t>(roundf(static_cast<float32_t>(sizeof(DataT))
                                                    / static_cast<float32_t>(sizeof(uint32_t))
                                                    * static_cast<float32_t>(Base::mM * Base::mN)))
                       / Base::mTBlockX);

            return x;
        }

        dim3 blockDim() const final
        {
            return dim3(Base::mTBlockX);
        }

        bool checkSizes() const final
        {
            return (Base::mTBlockY == 1);
        }

        bool checkDevice() const final
        {
            auto& deviceInfo = Base::DeviceInfo::instance();

            auto deviceArch = deviceInfo->getGcnArch();
            auto warpSize   = deviceInfo->warpSize();

            // gfx908 doesn't support dpp BCast16
            bool dppBCast16Check
                = !((deviceArch == Base::DeviceInfo::GFX908)
                    && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                    && (CrossLaneOp::groupSize() == 16u));

            // gfx11 doesn't support dpp wave shift / rotate, bcast15x16 or bcast31x32
            bool isGfx11 = (deviceArch == Base::DeviceInfo::GFX1100
                            || deviceArch == Base::DeviceInfo::GFX1101
                            || deviceArch == Base::DeviceInfo::GFX1102);

            bool dppWaveShiftCheck
                = !(isGfx11 && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_SHIFT)
                    && (CrossLaneOp::groupSize() == warpSize));

            bool dppWaveRotateCheck
                = !(isGfx11 && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                    && (CrossLaneOp::groupSize() == warpSize));

            bool dppWaterfallBCastCheck
                = !(isGfx11 && (CrossLaneOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_DPP)
                    && (CrossLaneOp::opId() == CrossLaneOps::Properties::OP_ID_WFALL_BCAST)
                    && (CrossLaneOp::groupSize() == warpSize));

            return Base::checkDevice() && dppBCast16Check && dppWaveShiftCheck && dppWaveRotateCheck
                   && dppWaterfallBCastCheck;
        }

        std::ostream& printHeader(std::ostream& stream = std::cout) const final
        {
            return stream << "DataT,"
                          << "Op_Id, "
                          << "Op_Impl, "
                          << "Op_Ctrl, "
                          << "Wave_Size, "
                          << "Group_Size, "
                          << "WriteRowMask, "
                          << "WriteBankMask, "
                          << "BoundCtrl, "
                          << "Result" << std::endl;
        }
        std::ostream& printKernel(std::ostream& stream = std::cout) const final
        {
            std::ios_base::fmtflags f(stream.flags()); // save flags state
            stream << dataTypeToString<DataT>() << ", " << CrossLaneOp::opId() << ", "
                   << CrossLaneOp::opImpl() << ", ";
            stream << std::showbase << std::hex << CrossLaneOp::opCtrl() << ", ";
            stream.flags(f);
            stream << "w" << CrossLaneOp::waveSize() << ", " << CrossLaneOp::groupSize() << ", ";
            stream << std::showbase << std::hex << WriteRowMask << ", " << WriteBankMask << ", ";
            stream.flags(f);
            stream << BoundCtrl << ", ";

            if(!Base::mRunFlag)
            {
                stream << "SKIPPED" << std::endl;
            }
            else
            {
                stream << (Base::mValidationResult ? "PASSED" : "FAILED") << std::endl;
            }

            stream.flags(f);
            return stream;
        }

        virtual void setupImpl(typename Base::DataStorage::ProblemSize const& probsize)
        {
            // Allocate storage
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            // Init host
            MatrixUtil<Layout>::fill(dataInstance->hostIn().get(), Base::mM, Base::mN);
            // for(int i=0; i < Base::mM * Base::mN; i++)
            // {
            //     dataInstance->hostIn().get()[i] = i;
            // }
            MatrixUtil<Layout>::fill(dataInstance->hostOut().get(),
                                     Base::mM,
                                     Base::mN,
                                     std::numeric_limits<DataT>::signaling_NaN());

            // Init device
            dataInstance->copyData(
                dataInstance->deviceIn(), dataInstance->hostIn(), Base::mM * Base::mN);
            dataInstance->copyData(
                dataInstance->deviceOut(), dataInstance->hostOut(), Base::mM * Base::mN);
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Perform the host validation
            // Host reference result in hostOut
            cross_lane_ref_dispatch_CPU<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>(
                dataInstance->hostOut().get(),
                dataInstance->hostIn().get(),
                Base::mM * Base::mN,
                Base::mParam1);

            // Copy back the GPU version.
            // GPU result in hostIn
            dataInstance->copyData(
                dataInstance->hostIn(), dataInstance->deviceOut(), Base::mM * Base::mN);

            // Compare on the host
            std::tie(Base::mValidationResult, Base::mMaxRelativeError)
                = compareEqual<DataT, DataT, Layout, Layout>(dataInstance->hostOut().get(),
                                                             dataInstance->hostIn().get(),
                                                             Base::mM,
                                                             Base::mN);

            // for(int i=0; i < Base::mM * Base::mN; i++)
            // {
            //     std::cout << std::hex;
            //     std::cout << "[" << dataInstance->hostOut().get()[i] << ", " << dataInstance->hostIn().get()[i] << "] " << std::endl;
            //     std::cout << std::dec;
            // }
        }

        virtual typename Base::KernelFunc kernelImpl() const = 0;
    };

    template <typename DataT, typename CrossLaneOp>
    struct BlendOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = CrossLaneOpsKernelBase<DataT, CrossLaneOp>;

    public:
        BlendOpsKernel()  = default;
        ~BlendOpsKernel() = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Allocate storage
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            // hostOut = src0
            // hostIn = src1
            MatrixUtil<typename Base::Layout>::fill(
                dataInstance->hostIn().get(), Base::mM, Base::mN);
            MatrixUtil<typename Base::Layout>::fill(
                dataInstance->hostOut().get(), Base::mM, Base::mN);

            // Mix things up so we don't test the same array.
            std::srand(std::time(nullptr));
            for(int i = 0; i < (Base::mM * Base::mN) / 3u; i++)
            {
                uint32_t swapL                      = uint32_t(std::rand()) % (Base::mM * Base::mN);
                uint32_t swapR                      = uint32_t(std::rand()) % (Base::mM * Base::mN);
                auto     tmp                        = dataInstance->hostIn().get()[swapL];
                dataInstance->hostIn().get()[swapL] = dataInstance->hostIn().get()[swapR];
                dataInstance->hostIn().get()[swapR] = tmp;
            }

            // for(int i=0; i < Base::mM * Base::mN; i++)
            // {
            //     dataInstance->hostIn().get()[i] = i;
            //     dataInstance->hostOut().get()[i] = i | 0xFFFFFF00;

            //     std::cout << std::hex;
            //     std::cout << "[" << dataInstance->hostOut().get()[i] << ", " << dataInstance->hostIn().get()[i] << "] " << std::endl;
            //     std::cout << std::dec;
            // }

            // Init device
            dataInstance->copyData(
                dataInstance->deviceIn(), dataInstance->hostIn(), Base::mM * Base::mN);
            dataInstance->copyData(
                dataInstance->deviceOut(), dataInstance->hostOut(), Base::mM * Base::mN);
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(blendOpsTest<DataT, CrossLaneOp>);
        }

        typename Base::DataStorage::HostPtrT mSrc1;
    };

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    struct DppOpsKernel final
        : public CrossLaneOpsKernelBase<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        DppOpsKernel()  = default;
        ~DppOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(
                dppOpsTest<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>);
        }
    };

    template <typename DataT, typename CrossLaneOp>
    struct SwizzleOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        SwizzleOpsKernel()  = default;
        ~SwizzleOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(swizzleOpsTest<DataT, CrossLaneOp>);
        }
    };

    template <typename DataT, typename CrossLaneOp>
    struct PermuteOpsKernel final : public CrossLaneOpsKernelBase<DataT, CrossLaneOp>
    {
        using Base = UnitKernelBase<1, 1, DataT, col_major>;

    public:
        PermuteOpsKernel()  = default;
        ~PermuteOpsKernel() = default;

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(permuteOpsTest<DataT, CrossLaneOp>);
        }
    };

    // This is the GeneratorImpl class
    struct DppOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT         = 0,
            CrossLaneOp   = 1,
            WriteRowMask  = 2,
            WriteBankMask = 3,
            BoundCtrl     = 4,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = DppOpsKernel<
                std::tuple_element_t<DataT, TestParamsT>, // DataT
                std::tuple_element_t<CrossLaneOp, TestParamsT>, // CrossLaneOp
                std::tuple_element_t<WriteRowMask, TestParamsT>::value, // WriteRowMask
                std::tuple_element_t<WriteBankMask, TestParamsT>::value, // WriteBankMask
                std::tuple_element_t<BoundCtrl, TestParamsT>::value // BoundCtrl
                >;

            return std::make_shared<KernelT>();
        }
    };

    struct SwizzleOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = SwizzleOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                   std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                   >;

            return std::make_shared<KernelT>();
        }
    };

    struct PermuteOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = PermuteOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                   std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                   >;

            return std::make_shared<KernelT>();
        }
    };

    struct BlendOpsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT       = 0,
            CrossLaneOp = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = BlendOpsKernel<std::tuple_element_t<DataT, TestParamsT>, // DataT
                                 std::tuple_element_t<CrossLaneOp, TestParamsT> // CrossLaneOp
                                 >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_CROSS_LANE_OPS_HPP
