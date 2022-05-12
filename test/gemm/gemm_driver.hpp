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
#ifndef GEMM_DRIVER_HPP
#define GEMM_DRIVER_HPP

namespace rocwmma
{
    namespace CooperativeGemm
    {
        template <typename GlobalMapping,
                typename LdsMapping,
                typename CoopSchedulerA,
                typename CoopSchedulerB>
        struct GemmDriver
        {
            // Global fragment types
            using GRFragA = typename GlobalMapping::GRFragA;
            using GRFragB = typename GlobalMapping::GRFragB;

            // Mfma fragment types
            using MfmaFragA = typename GlobalMapping::MfmaFragA;
            using MfmaFragB = typename GlobalMapping::MfmaFragB;
            using MfmaFragC = typename GlobalMapping::MfmaFragC;
            using MfmaFragD = typename GlobalMapping::MfmaFragD;
            using MfmaFragAcc = typename GlobalMapping::MfmaFragAcc;

            // Local fragment types
            using LWFragA = typename LdsMapping::LWFragA;
            using LWFragB = typename LdsMapping::LWFragB;
            using LRFragA = typename LdsMapping::LRFragA;
            using LRFragB = typename LdsMapping::LRFragB;

            template<typename FragT>
            using IOTraits = typename FragT::IOConfig::IOTraits;

            template<typename FragT>
            using DataT = typename FragT::element_type;

            template<typename FragT>
            using MappingUtil = typename FragT::IOConfig::MappingUtil;

            // Ensure that splitCounts are the same on both sides of
            // global fetch and local writes to match fragment data locality.
            constexpr static auto splitCount = std::min((uint32_t)IOTraits<GRFragB>::IOCount,
                                                        (uint32_t)IOTraits<LWFragB>::IOCount);

            static_assert(((uint32_t)IOTraits<GRFragB>::IOCount % splitCount == 0u) &&
                        ((uint32_t)IOTraits<LWFragB>::IOCount % splitCount == 0u),
                            "splitCount is not common divisor of GlobalRead and LocalWrite IOCounts");

            ///
            /// Broadcast (fill) value
            ///

            // Broadcast value to fragment
            // Single or BlocksX * BlocksY frags
            template<typename FragT>
            __device__ static inline void fill(FragT& frag, DataT<FragT> value);
            template<typename FragT, uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void fill(FragT (&frags)[BlocksX][BlocksY], DataT<FragT> value);

            ///
            /// Global R/W
            ///

            // Global A/B reads in cooperative mode
            __device__ static inline void globalReadCoopA(GRFragA& grFragA, DataT<GRFragA> const* gAddrA, uint32_t lda);
            __device__ static inline void globalReadCoopB(GRFragB& grFragB, DataT<GRFragB> const* gAddrB, uint32_t ldb);

            // Global C reads non-cooperative
            // Single or BlocksX * BlocksY frags
            template<uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void globalReadC(MfmaFragC (&fragC)[BlocksX][BlocksY], DataT<MfmaFragC> const* gAddrC, uint32_t ldc);
            __device__ static inline void globalReadC(MfmaFragC& fragC, DataT<MfmaFragC> const* gAddrC, uint32_t ldc);
            
            // Global D writes non-cooperative
            // Single or BlocksX * BlocksY frags
            template<uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void globalWriteD(DataT<MfmaFragD>* gAddrD, MfmaFragD const (&fragsD)[BlocksX][BlocksY], uint32_t ldd);
            __device__ static inline void globalWriteD(DataT<MfmaFragD>* gAddrD, MfmaFragD const& fragD, uint32_t ldd);

            ///
            /// Local R/W
            ///

            // Local A/B writes in cooperative mode
            __device__ static inline void localWriteCoopA(DataT<GRFragA>* ldsAddr, GRFragA const& grFragA, uint32_t ldlds);
            __device__ static inline void localWriteCoopB(DataT<GRFragB>* ldsAddr, GRFragB const& grFragB, uint32_t ldlds);

            // Local A read non-cooperative
            // Single or BlocksX frags
            template<uint32_t BlocksX>
            __device__ static inline void localReadA(MfmaFragA (&fragsA)[BlocksX], DataT<MfmaFragA> const* ldsAddrA, uint32_t ldlds);
            __device__ static inline void localReadA(MfmaFragA& fragsA, DataT<MfmaFragA> const* ldsAddrA, uint32_t ldlds);
            
            // Local B read non-cooperative
            // Single or BlocksY frags
            template<uint32_t BlocksY>
            __device__ static inline void localReadB(MfmaFragB (&fragsB)[BlocksY], DataT<MfmaFragB> const* ldsAddrB, uint32_t ldlds);
            __device__ static inline void localReadB(MfmaFragB& fragsB, DataT<MfmaFragB> const* ldsAddrB, uint32_t ldlds);

            ///
            /// MFMA
            ///

            // Performs mfma
            // Single block, or BlocksX * BlocksY frags
            template<uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void mfma(MfmaFragAcc (&fragAccOut)[BlocksX][BlocksY], MfmaFragA const (&fragA)[BlocksX], MfmaFragB const (&fragB)[BlocksY], MfmaFragAcc const (&fragAccIn)[BlocksX][BlocksY]);
            __device__ static inline void mfma(MfmaFragAcc& fragAccOut, MfmaFragA const& fragA, MfmaFragB const& fragB, MfmaFragAcc const& fragAccIn);

            ///
            /// Uniform fused multiply - add (FMA)
            ///

            // Performs D = alpha * acc + beta * C, where alpha, beta are uniform scalars
            template<uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void uniformFma(MfmaFragD (&fragsD)[BlocksX][BlocksY],
                                                    DataT<MfmaFragAcc> alpha, MfmaFragAcc const (&fragsAcc)[BlocksX][BlocksY],
                                                    DataT<MfmaFragAcc> beta,  MfmaFragC const (&fragsC)[BlocksX][BlocksY]);
            __device__ static inline void uniformFma(MfmaFragD& fragD, DataT<MfmaFragAcc> alpha, MfmaFragAcc const& fragAcc, DataT<MfmaFragAcc> beta, MfmaFragC const& fragC);

            ///
            /// Wave synchronization
            ///
            __device__ static inline void syncWorkgroup();
        };

    } // namespace CooperativeGemm

} // namespace rocwmma

#include "gemm_driver_impl.hpp"

#endif // GEMM_DRIVER_HPP
