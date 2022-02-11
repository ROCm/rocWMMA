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

#ifndef DLRM_DOT_LDS_DETAIL_H
#define DLRM_DOT_LDS_DETAIL_H

#include "DlrmKernelBase.h"
#include "device/DlrmDotBwdLds.h"
#include "device/DlrmDotFwdLds.h"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t TileSize, typename DataT, typename MappingLds>
    struct DlrmDotLdsKernel final : public DlrmKernelBase<TileSize, DataT>
    {
    private:
        using Base = DlrmKernelBase<TileSize, DataT>;

    public:
        DlrmDotLdsKernel() {}
        ~DlrmDotLdsKernel() final {}

        uint32_t ldsUsage() const final
        {
            auto blockDims = this->blockDim();
            return sizeof(DataT) * (blockDims.x / AMDGCN_WAVE_SIZE * blockDims.y) * 2
                   * (TileSize * TileSize);
        }
        typename Base::KernelFwdFunc kernelFwdImpl() const final
        {
            return typename Base::KernelFwdFunc(dlrmDotFwdLds<DataT, TileSize, MappingLds>);
        }

        typename Base::KernelBwdFunc kernelBwdImpl() const final
        {
            return typename Base::KernelBwdFunc(dlrmDotBwdLds<DataT, TileSize, MappingLds>);
        }

        typename Base::KernelTrilFunc kernelTrilImpl() const final
        {
            return typename Base::KernelTrilFunc(trilReconstructLds<DataT>);
        }
    };

    // This is the GeneratorImpl class
    struct DlrmDotLdsGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT      = 0,
            TileSize   = 1,
            MappingLds = 2
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = DlrmDotLdsKernel<std::tuple_element_t<TileSize, TestParamsT>::value,
                                             std::tuple_element_t<DataT, TestParamsT>,
                                             std::tuple_element_t<MappingLds, TestParamsT>>;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // DLRM_DOT_LDS_DETAIL_H
