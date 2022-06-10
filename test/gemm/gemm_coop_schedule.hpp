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
#ifndef GEMM_COOP_SCHEDULE_HPP
#define GEMM_COOP_SCHEDULE_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace CooperativeGemm
    {
        namespace Schedule
        {
            // Collaborative waves in the same workgroup row only.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)  => Share Schedule: i0 = (0, 0), i1 = (0, 1), count = 2
            // (1, 0)   (1, 1)  => Share Schedule: i0 = (1, 0), i2 = (1, 1), count = 2
            struct SameRowFwd
            {
                constexpr static inline auto waveIndex()
                {
                    return std::get<1>(detail::WaveSpace::localWaveCoord());
                }
                constexpr static inline auto waveCount()
                {
                    return std::get<1>(detail::WaveSpace::workgroupDim());
                }
            };

            // Collaborative waves in the same workgroup col only.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)
            // (1, 0)   (1, 1)
            //   ||       ||
            //   ||       \/
            //   \/    Share Schedule: i0 = (0, 1), i1 = (1, 1), count = 2
            // Share Schedule: i0 = (0, 1), i1 = (1, 0), count = 2
            struct SameColFwd
            {
                constexpr static inline auto waveIndex()
                {
                    return std::get<0>(detail::WaveSpace::localWaveCoord());
                }
                constexpr static inline auto waveCount()
                {
                    return std::get<0>(detail::WaveSpace::workgroupDim());
                }
            };

            // All waves are collaborative.
            // Scheduling order is analogous to row major priority.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)   Share Schedule: i0 = (0, 0), i1 = (0, 1),
            // (1, 0)   (1, 1)                   i2 = (1, 0), i3 = (1, 1), count = 4
            struct AllRowMajor
            {
                constexpr static inline auto waveIndex()
                {
                    auto localWaveCoord = detail::WaveSpace::localWaveCoord();
                    return std::get<0>(localWaveCoord)
                               * std::get<1>(detail::WaveSpace::workgroupDim())
                           + std::get<1>(localWaveCoord);
                }
                constexpr static inline auto waveCount()
                {
                    auto wgDim = detail::WaveSpace::workgroupDim();
                    return std::get<0>(wgDim) * std::get<1>(wgDim);
                }
            };

            // All waves are collaborative.
            // Scheduling order is analogous to col major priority.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)   Share Schedule: i0 = (0, 0), i2 = (0, 1),
            // (1, 0)   (1, 1)                   i1 = (1, 0), i3 = (1, 1), count = 4
            struct AllColMajor
            {
                constexpr static inline auto waveIndex()
                {
                    auto localWaveCoord = detail::WaveSpace::localWaveCoord();
                    return std::get<1>(localWaveCoord)
                               * std::get<0>(detail::WaveSpace::workgroupDim())
                           + std::get<0>(localWaveCoord);
                }
                constexpr static inline auto waveCount()
                {
                    auto wgDim = detail::WaveSpace::workgroupDim();
                    return std::get<0>(wgDim) * std::get<1>(wgDim);
                }
            };

        } // namespace Schedule

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_COOP_SCHEDULE_HPP
