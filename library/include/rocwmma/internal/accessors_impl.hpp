/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_ACCESSORS_IMPL_HPP
#define ROCWMMA_ACCESSORS_IMPL_HPP

#include "accessors.hpp"
#include "api_fwd.hpp"
#include "coop_io_config.hpp"
#include "fragment_traits.hpp"
#include "io_config.hpp"
#include "io_scheduler.hpp"
#include "io_shape.hpp"
#include "mma_config.hpp"

namespace rocwmma
{
    template <typename FragT>
    struct GetIOConfig;

    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct GetIOConfig<fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>>
    {
        using type = CoopIOConfig<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>;
    };

    template <typename FragA, typename FragB, typename FragC, typename FragD>
    struct GetMmaConfig
    {
    private:
        using FragATraits = fragment_traits<FragA>;
        using FragBTraits = fragment_traits<FragB>;
        using FragCTraits = fragment_traits<FragC>;
        using FragDTraits = fragment_traits<FragD>;

        // sanity checks
        static_assert((FragATraits::FragM == FragBTraits::FragM)
                          && (FragBTraits::FragM == FragCTraits::FragM)
                          && (FragCTraits::FragM == FragDTraits::FragM),
                      "Mma fragment FragM traits must match");
        static_assert((FragATraits::FragN == FragBTraits::FragN)
                          && (FragBTraits::FragN == FragCTraits::FragN)
                          && (FragCTraits::FragN == FragDTraits::FragN),
                      "Mma fragment FragN traits must match");
        static_assert((FragATraits::FragK == FragBTraits::FragK)
                          && (FragBTraits::FragK == FragCTraits::FragK)
                          && (FragCTraits::FragK == FragDTraits::FragK),
                      "Mma fragment FragK traits must match");
        static_assert(is_same_v<typename FragCTraits::DataT, typename FragDTraits::DataT>,
                      "Accum fragments C and D must have the same type");

        static_assert(
            is_same_v<
                typename FragATraits::Scheduler,
                typename FragBTraits::
                    Scheduler> && is_same_v<typename FragBTraits::Scheduler, typename FragCTraits::Scheduler> && is_same_v<typename FragCTraits::Scheduler, typename FragDTraits::Scheduler>,
            "Mma fragment scheduler traits must match");

        static_assert(!scheduler_traits<typename FragATraits::Scheduler>::is_cooperative,
                      "Mma does not support cooperative fragments");

    public:
        // We've already checked:
        // - FragMNK for all frags match
        // - DataT match for C and D
        using type = MmaConfig<FragATraits::FragM,
                               FragATraits::FragN,
                               FragATraits::FragK,
                               typename FragATraits::DataT,
                               typename FragBTraits::DataT,
                               typename FragCTraits::DataT,
                               typename FragATraits::DataLayoutT,
                               typename FragBTraits::DataLayoutT,
                               typename FragCTraits::DataLayoutT,
                               typename FragDTraits::DataLayoutT>;
    };

    ///
    /// CoopConfig access
    ///

    template <typename FragT, uint32_t WaveCount>
    struct GetCoopIOConfig;

    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler,
              uint32_t WaveCount>
    struct GetCoopIOConfig<fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>,
                           WaveCount>
    {
        using type = CoopIOConfig<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>;
    };

    ///
    /// IOShape access
    ///

    template <typename FragT>
    struct GetIOShape;

    template <typename MatrixT, uint32_t FragM, uint32_t FragN, uint32_t FragK, typename... Ts>
    struct GetIOShape<fragment<MatrixT, FragM, FragN, FragK, Ts...>>
    {
        using type = IOShape<MatrixT, FragM, FragN, FragK>;
    };

    // template <typename MatrixT,
    //           uint32_t BlockM,
    //           uint32_t BlockN,
    //           uint32_t BlockK,
    //           typename DataT,
    //           typename DataLayoutT>
    // struct GetDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    // {
    //     using type = DataLayout::template Array1d<DataLayoutT>;
    // };

    template <typename FragT>
    struct GetMappingUtil;

    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename... Ts>
    struct GetMappingUtil<fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Ts...>>
    {
    private:
        using IOShapeT = IOShape<MatrixT, FragM, FragN, FragK>;

    public:
        using type = MappingUtil<IOShapeT::BlockHeight, IOShapeT::BlockWidth, DataT, DataLayoutT>;
    };

} // namespace rocwmma

#endif // ROCWMMA_ACCESSORS_IMPL_HPP
