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

#ifndef DLRM_TEST_PARAMS_HPP
#define DLRM_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <rocwmma/internal/types.hpp>

#include "../common.hpp"
#include "KernelGenerator.h"
#include "dlrm_kernel_base.hpp"

namespace rocwmma
{

    struct DlrmTestParams
    {
        // Types of parameters
        using KernelT        = std::shared_ptr<KernelI>;
        using ThreadBlockT   = std::pair<int64_t, int64_t>;
        using ProblemSizeT   = std::tuple<int64_t, int64_t, int64_t>;
        using PassDirectionT = DlrmDirection_t;

        using DataTypes = std::tuple<std::tuple<float32_t>, std::tuple<float16_t>>;
        using TileSizes = std::tuple<std::tuple<I<16>>, std::tuple<I<32>>>;

        // M, K, BatchSize
        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return {{32, 32, 64}, {32, 128, 64}, {128, 128, 64}, {1024, 1024, 64}};
        }
        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            return {{128, 1}};
        }

        static inline std::vector<PassDirectionT> passDirections()
        {
            return {DlrmDirection_t::Forward, DlrmDirection_t::Backward};
        }
    };

} // namespace rocwmma

#endif // DLRM_TEST_PARAMS_HPP
