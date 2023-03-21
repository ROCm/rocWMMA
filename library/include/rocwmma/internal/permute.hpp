/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_PERMUTE_HPP
#define ROCWMMA_PERMUTE_HPP

#include "cross_lane_ops.hpp"
#include "permute_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{
    namespace Permute
    {

        template <typename PermuteOp>
        struct Driver
        {
        private:
            template <typename DataT, uint32_t VecSize, uint32_t... Idx>
            ROCWMMA_DEVICE static inline auto
                forEach(VecT<DataT, VecSize> const& src, uint32_t laneId, detail::SeqT<Idx...>)
            {
                static_assert(sizeof...(Idx) == VecSize, "Index count must match vector size");
                return VecT<DataT, VecSize>{PermuteOp::exec(get<Idx>(src), laneId)...};
            }

        public:
            // Sanity checks
            static_assert((PermuteOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_PERMUTE)
                              || (PermuteOp::opImpl()
                                  == CrossLaneOps::Properties::OP_IMPL_BPERMUTE),
                          "PermuteOp must use permute or permute backend");
            static_assert((PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_BLOCK_BCAST)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_GATHER)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_SCATTER),
                          "PermuteOp is unsupported");

            template <typename DataT>
            ROCWMMA_DEVICE static inline auto exec(DataT const& src)
            {
                return PermuteOp::exec(src, detail::WaveSpace<>::localLaneId());
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src)
            {
// TODO: Investigate static unroll validation
#if ROCWMMA_ARCH_GFX1102
                VecT<DataT, VecSize> result;
                auto                 itW = makeVectorIterator(result).begin();
                auto const           itR = makeVectorIterator(src).begin();

                static_assert(decltype(itR)::range() == VecSize,
                              "VecSize inconsistent with iterator range");
                static_assert(decltype(itW)::range() == VecSize,
                              "VecSize inconsistent with iterator range");

#pragma unroll
                for(uint32_t i = 0; i < VecSize; ++i, itR++, itW++)
                {
                    get<0>(*itW) = exec(get<0>(*itR));
                }

                return result;
#else

                return forEach(src, detail::WaveSpace<>::localLaneId(), detail::Seq<VecSize>{});
#endif // ROCWMMA_ARCH_GFX1102
            }
        };

        template <uint32_t BlockIdx>
        using BlockBCast32 = Driver<PermuteImpl::Ops::BlockBCast32<BlockIdx>>;

        template <uint32_t BlockIdx>
        using BlockBCast16 = Driver<PermuteImpl::Ops::BlockBCast16<BlockIdx>>;

        template <uint32_t BlockIdx>
        using BlockBCast8 = Driver<PermuteImpl::Ops::BlockBCast8<BlockIdx>>;

        template <uint32_t BlockIdx>
        using BlockBCast4 = Driver<PermuteImpl::Ops::BlockBCast4<BlockIdx>>;

        template <uint32_t BlockIdx>
        using BlockBCast2 = Driver<PermuteImpl::Ops::BlockBCast2<BlockIdx>>;

        template <uint32_t VW, uint32_t ElementShift>
        using GatherWave = Driver<PermuteImpl::Ops::GatherWave<VW, ElementShift>>;

        template <uint32_t VW, uint32_t ElementShift>
        using Gather32 = Driver<PermuteImpl::Ops::Gather32<VW, ElementShift>>;

        template <uint32_t VW, uint32_t ElementShift>
        using Gather16 = Driver<PermuteImpl::Ops::Gather16<VW, ElementShift>>;

        template <uint32_t VW, uint32_t ElementShift>
        using ScatterWave = Driver<PermuteImpl::Ops::ScatterWave<VW, ElementShift>>;

        template <uint32_t VW, uint32_t ElementShift>
        using Scatter32 = Driver<PermuteImpl::Ops::Scatter32<VW, ElementShift>>;

        template <uint32_t VW, uint32_t ElementShift>
        using Scatter16 = Driver<PermuteImpl::Ops::Scatter16<VW, ElementShift>>;

        template <uint32_t RotateDistance>
        using RotateWaveL = Driver<PermuteImpl::Ops::RotateWaveL<RotateDistance>>;

        template <uint32_t RotateDistance>
        using RotateWaveR = Driver<PermuteImpl::Ops::RotateWaveR<RotateDistance>>;

    } // namespace Permute

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_HPP
