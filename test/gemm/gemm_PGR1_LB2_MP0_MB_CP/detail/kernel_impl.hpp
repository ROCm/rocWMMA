/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_GEMM_TEST_DETAIL_KERNEL
#define ROCWMMA_GEMM_TEST_DETAIL_KERNEL

#include "device/kernel_device_func.hpp"
#include "gemm_kernel_base.hpp"
#include "helper_macros.hpp"

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
    struct Kernel_PGR1_LB2_MP0_MB_CP final : public GemmKernelBase<BlockM,
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

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveSize, uint32_t ArchId>
        using TestGuard = gemm_PGR1_LB2_MP0_MB_CP_guard<BlockM,
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
                                                        BlocksX,
                                                        BlocksY,
                                                        TBlockX,
                                                        TBlockY,
                                                        WaveSize,
                                                        ArchId>;

    public:
        Kernel_PGR1_LB2_MP0_MB_CP() {}
        ~Kernel_PGR1_LB2_MP0_MB_CP() final {}

        dim3 gridDim() const final
        {
            return dim3(ceilDiv(Base::mM,
                                BlockM * BlocksX * Base::mTBlockX
                                    / Base::DeviceInfo::instance()->warpSize()),
                        ceilDiv(Base::mN, BlockN * BlocksY * Base::mTBlockY));
        }

        bool checkSizes() const final
        {
            return ((BlockM * BlocksX * Base::mTBlockX / Base::DeviceInfo::instance()->warpSize())
                    <= Base::mM)
                   && ((BlockN * BlocksY * Base::mTBlockY) <= Base::mN) && (BlockK <= Base::mK);
        }

        bool checkQuirks() const final
        {
            auto waveSize   = Base::DeviceInfo::instance()->warpSize();
            auto deviceArch = Base::DeviceInfo::instance()->getGcnArch();

            // The test guard for this class requires 4 values at runtime.
            auto dispatchGuard = [waveSize, deviceArch, this]() {
                bool dispatchResult = false;

#define CASE_IMPL_ASSIGN4(TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    dispatchResult = TestGuard<TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID>::enable();

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY4_ARG4(                             \
        Base::mTBlockX, CASE_IMPL_ASSIGN4, 32u, 64u, 128u, 256u, TBLOCK_Y, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_TBLOCK_Y(WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY3_ARG3(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
    ROCWMMA_SWITCH_BODY2_ARG2(         \
        waveSize, SWITCH_BODY_TBLOCK_Y, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

#define DISPATCH_GUARD_BODY                          \
    ROCWMMA_SWITCH_BODY8_ARG1(deviceArch,            \
                              SWITCH_BODY_WAVE_SIZE, \
                              HipDevice::GFX908,     \
                              HipDevice::GFX90A,     \
                              HipDevice::GFX940,     \
                              HipDevice::GFX941,     \
                              HipDevice::GFX942,     \
                              HipDevice::GFX1100,    \
                              HipDevice::GFX1101,    \
                              HipDevice::GFX1102)

                DISPATCH_GUARD_BODY

#undef CASE_IMPL_ASSIGN4
#undef SWITCH_BODY_TBLOCK_X
#undef SWITCH_BODY_TBLOCK_Y
#undef SWITCH_BODY_WAVE_SIZE
#undef DISPATCH_GUARD_BODY

                return dispatchResult;
            };

            auto dispatchCheck = dispatchGuard();

            // Don't run the kernel if the threadblock size is not supported
            auto kernelImplCheck = (kernelImpl() != nullptr);

            // Cooperative workgroup kernels quirks
            auto wgQuirksCheck = true;
            if(std::is_same<GemmConfig, CooperativeGemm::WorkgroupLevel::LdsNT>::value
               || std::is_same<GemmConfig, CooperativeGemm::WorkgroupLevel::LdsTN>::value)
            {
                // TODO: Fp64 fails validation for BlockK > 16 for 16 x 16.
                wgQuirksCheck &= !(std::is_same<InputT, float64_t>::value && (BlockM == 16)
                                   && (BlockN == 16) && (BlockK > 16) && (BlocksX * BlocksY >= 16));
            }

            // Cooperative wave kernels quirks
            auto waveQuirksCheck = true;
            if(std::is_same<GemmConfig, CooperativeGemm::WaveLevel::LdsNT>::value
               || std::is_same<GemmConfig, CooperativeGemm::WaveLevel::LdsTN>::value)
            {
                // TODO: On gfx90a, TN config with 4x4 blocks of 32 x 32 x 8
                // Produces compile time issues
                waveQuirksCheck &= !((deviceArch == Base::DeviceInfo::GFX90A) && // GFX90A
                                     (std::is_same<LayoutA, row_major>::value
                                      && std::is_same<LayoutB, col_major>::value)
                                     && // TN config
                                     ((BlockM == BlockN == 32) && BlockK == 8) && // 32 x 32 x 8
                                     (BlocksX == BlocksY == 4)); // BlocksX = 4, BlocksY = 4
            }

            return Base::checkQuirks() && dispatchCheck && kernelImplCheck && wgQuirksCheck
                   && waveQuirksCheck;
        }

        // Lds memory usage in bytes
        uint32_t ldsUsage() const final
        {
            // Uses 2 lds blocks for prefetch loop
            return 2 * sizeof(InputT)
                   * (Base::mTBlockX / Base::DeviceInfo::instance()->warpSize() * BlocksX * BlockM
                      + Base::mTBlockY * BlocksY * BlockN)
                   * BlockK;
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            // Runtime dispatcher to assign compile time TBlock params.
            auto kernelFunc = typename Base::KernelFunc(nullptr);

            // The test guard for this class requires 4 values at runtime.
            //
            auto dispatchKernelFunc = [&kernelFunc, this]() {
#define CASE_IMPL_ASSIGN2(TBLOCK_X, TBLOCK_Y)                                  \
    kernelFunc = typename Base::KernelFunc(gemm_PGR1_LB2_MP0_MB_CP<BlockM,     \
                                                                   BlockN,     \
                                                                   BlockK,     \
                                                                   InputT,     \
                                                                   OutputT,    \
                                                                   ComputeT,   \
                                                                   LayoutA,    \
                                                                   LayoutB,    \
                                                                   LayoutC,    \
                                                                   LayoutD,    \
                                                                   LayoutLds,  \
                                                                   GemmConfig, \
                                                                   BlocksX,    \
                                                                   BlocksY,    \
                                                                   TBLOCK_X,   \
                                                                   TBLOCK_Y>);

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y) \
    ROCWMMA_SWITCH_BODY4_ARG2(Base::mTBlockX, CASE_IMPL_ASSIGN2, 32u, 64u, 128u, 256u, TBLOCK_Y)

#define DISPATCH_KERNEL_FUNC_BODY \
    ROCWMMA_SWITCH_BODY3_ARG1(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u)
                DISPATCH_KERNEL_FUNC_BODY

#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_TBLOCK_X
#undef DISPATCH_KERNEL_FUNC_BODY
            };

            dispatchKernelFunc();

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

} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DETAIL_KERNEL
