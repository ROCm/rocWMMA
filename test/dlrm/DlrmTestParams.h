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

#ifndef DLRM_TEST_PARAMS_H
#define DLRM_TEST_PARAMS_H

#include <tuple>
#include <vector>

#include "Common.hpp"
#include "DlrmKernelBase.h"
#include "KernelGenerator.h"
#include "Types.h"

struct DlrmTestParams
{
    // Types of parameters
    using KernelT        = std::shared_ptr<KernelI>;
    using ThreadBlockT   = std::pair<int64_t, int64_t>;
    using ProblemSizeT   = std::tuple<int64_t, int64_t, int64_t>;
    using FwdDataSizeT   = std::tuple<int64_t, int64_t, int64_t>;
    using BwdDataSizeT   = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using PassDirectionT = bool;

    using DataTypes      = std::tuple<std::tuple<float32_t>, std::tuple<float16_t>>;
    using TestBlockSizes = std::tuple<std::tuple<I<16>, I<16>, I<16>>>;
    using TileSizes      = std::tuple<std::tuple<I<16>, I<32>>>;

    static inline std::vector<ProblemSizeT> problemSizes()
    {
        return {{64, 64, 64}};
    }
    static inline std::vector<ThreadBlockT> threadBlocks()
    {
        //return {{64, 1}, {64, 2}, {64, 4}, {128, 1}, {128, 2}, {256, 1}};
        return {{256, 1}};
    }

    static inline std::vector<FwdDataSizeT> fwdDataSizes()
    {
        return {{100, 100, 100}};
    }

    static inline std::vector<BwdDataSizeT> bwdDataSizes()
    {
        return {{100, 100, 100, 100, 100, 100}};
    }

    static inline std::vector<PassDirectionT> passDirections()
    {
        return {false, true};
    }
};

#endif // DLRM_TEST_PARAMS_H
