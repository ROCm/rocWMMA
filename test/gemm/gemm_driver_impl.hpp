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
#ifndef GEMM_DRIVER_IMPL_HPP
#define GEMM_DRIVER_IMPL_HPP

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

#define GemmDriverT \
    typename GlobalMapping, typename LdsMapping, typename CoopSchedulerA, typename CoopSchedulerB

#define GemmDriverT_impl GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopA(
            GRFragA& grFragA, DataT<GRFragA> const* gAddrA, uint32_t lda)
        {
            rocwmma::load_matrix_coop_sync(grFragA,
                                           gAddrA,
                                           lda,
                                           CoopSchedulerA::waveIndex(),
                                           CoopSchedulerA::waveCount(),
                                           splitCountA);
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopB(
            GRFragB& grFragB, DataT<GRFragB> const* gAddrB, uint32_t ldb)
        {
            rocwmma::load_matrix_coop_sync(grFragB,
                                           gAddrB,
                                           ldb,
                                           CoopSchedulerB::waveIndex(),
                                           CoopSchedulerB::waveCount(),
                                           splitCountB);
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopA(
            DataT<GRFragA>* ldsAddr, GRFragA const& grFragA, uint32_t ldlds)
        {
            rocwmma::store_matrix_coop_sync(ldsAddr,
                                            reinterpret_cast<LWFragA const&>(grFragA),
                                            ldlds,
                                            CoopSchedulerA::waveIndex(),
                                            CoopSchedulerA::waveCount(),
                                            splitCountA);
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopB(
            DataT<GRFragB>* ldsAddr, GRFragB const& grFragB, uint32_t ldlds)
        {
            rocwmma::store_matrix_coop_sync(ldsAddr,
                                            reinterpret_cast<LWFragB const&>(grFragB),
                                            ldlds,
                                            CoopSchedulerB::waveIndex(),
                                            CoopSchedulerB::waveCount(),
                                            splitCountB);
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadA(
            MfmaFragA& fragsA, DataT<MfmaFragA> const* ldsAddrA, uint32_t ldlds)
        {
            rocwmma::load_matrix_sync(reinterpret_cast<LRFragA&>(fragsA), ldsAddrA, ldlds);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadA(
            MfmaFragA (&fragsA)[BlocksX], DataT<MfmaFragA> const* ldsAddrA, uint32_t ldlds)
        {
            auto blockStep = MappingUtil<LRFragA>::dataOffset(LdsMapping::blockOffsetA(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                localReadA(fragsA[i], ldsAddrA, ldlds);
                ldsAddrA += blockStep;
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadB(
            MfmaFragB& fragsB, DataT<MfmaFragB> const* ldsAddrB, uint32_t ldlds)
        {
            rocwmma::load_matrix_sync(reinterpret_cast<LRFragB&>(fragsB), ldsAddrB, ldlds);
        }

        template <GemmDriverT>
        template <uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadB(
            MfmaFragB (&fragsB)[BlocksY], DataT<MfmaFragB> const* ldsAddrB, uint32_t ldlds)
        {
            auto blockStep = MappingUtil<LRFragB>::dataOffset(LdsMapping::blockOffsetB(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                localReadB(fragsB[i], ldsAddrB, ldlds);
                ldsAddrB += blockStep;
            }
        }

        template <GemmDriverT>
        template <typename FragT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::fill(FragT& frag, DataT<FragT> value)
        {
            rocwmma::fill_fragment(frag, value);
        }

        template <GemmDriverT>
        template <typename FragT, uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::fill(FragT (&frags)[BlocksX][BlocksY],
                                                                  DataT<FragT> value)
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    fill(frags[i][j], value);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::mfma(MfmaFragAcc&       fragAccOut,
                                                                  MfmaFragA const&   fragA,
                                                                  MfmaFragB const&   fragB,
                                                                  MfmaFragAcc const& fragAccIn)
        {
            rocwmma::mma_sync(fragAccOut, fragA, fragB, fragAccIn);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void
            GemmDriver<GemmDriverT_impl>::mfma(MfmaFragAcc (&fragAccOut)[BlocksX][BlocksY],
                                               MfmaFragA const (&fragA)[BlocksX],
                                               MfmaFragB const (&fragB)[BlocksY],
                                               MfmaFragAcc const (&fragAccIn)[BlocksX][BlocksY])
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    mfma(fragAccOut[i][j], fragA[i], fragB[j], fragAccIn[i][j]);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadC(
            MfmaFragC& fragC, DataT<MfmaFragC> const* gAddrC, uint32_t ldc)
        {
            rocwmma::load_matrix_sync(fragC, gAddrC, ldc);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadC(
            MfmaFragC (&fragC)[BlocksX][BlocksY], DataT<MfmaFragC> const* gAddrC, uint32_t ldc)
        {
            auto blockStepX
                = MappingUtil<MfmaFragC>::dataOffset(GlobalMapping::blockOffsetA(), ldc);
            auto blockStepY
                = MappingUtil<MfmaFragC>::dataOffset(GlobalMapping::blockOffsetB(), ldc);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                auto offsetY = 0u;
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    globalReadC(fragC[i][j], gAddrC + offsetY, ldc);

                    offsetY += blockStepY;
                }
                gAddrC += blockStepX;
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalWriteD(DataT<MfmaFragD>* gAddrD,
                                                                          MfmaFragD const&  fragD,
                                                                          uint32_t          ldd)
        {
            rocwmma::store_matrix_sync(gAddrD, fragD, ldd);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalWriteD(
            DataT<MfmaFragD>* gAddrD, MfmaFragD const (&fragsD)[BlocksX][BlocksY], uint32_t ldd)
        {
            auto blockStepX
                = MappingUtil<MfmaFragD>::dataOffset(GlobalMapping::blockOffsetA(), ldd);
            auto blockStepY
                = MappingUtil<MfmaFragD>::dataOffset(GlobalMapping::blockOffsetB(), ldd);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                auto offsetY = 0u;
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    globalWriteD(gAddrD + offsetY, fragsD[i][j], ldd);

                    offsetY += blockStepY;
                }
                gAddrD += blockStepX;
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::uniformFma(MfmaFragD&         fragD,
                                                                        DataT<MfmaFragAcc> alpha,
                                                                        MfmaFragAcc const& fragAcc,
                                                                        DataT<MfmaFragAcc> beta,
                                                                        MfmaFragC const&   fragC)
        {
            for(int i = 0; i < fragD.num_elements; i++)
            {
                // Perform computation in ComputeT and cast back to OutputT
                fragD.x[i] = static_cast<DataT<MfmaFragD>>(
                    alpha * fragAcc.x[i] + beta * static_cast<DataT<MfmaFragAcc>>(fragC.x[i]));
            }
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::uniformFma(
            MfmaFragD (&fragsD)[BlocksX][BlocksY],
            DataT<MfmaFragAcc> alpha,
            MfmaFragAcc const (&fragsAcc)[BlocksX][BlocksY],
            DataT<MfmaFragAcc> beta,
            MfmaFragC const (&fragsC)[BlocksX][BlocksY])
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    uniformFma(fragsD[i][j], alpha, fragsAcc[i][j], beta, fragsC[i][j]);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::syncWorkgroup()
        {
            rocwmma::synchronize_workgroup();
        }

#undef GemmDriverT
#undef GemmDriverT_impl

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_DRIVER_IMPL_HPP
