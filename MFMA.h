#ifndef WMMA_MFMA_H
#define WMMA_MFMA_H

#include "Convert.h"
#include "IOPack.h"
#include "IOTraits.h"
#include "IOUnpack.h"
#include "Types.h"

template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN>
struct amdgcn_mfma;

template <>
struct amdgcn_mfma<float16_t, float32_t, 16, 16>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 16,
        };
        using ARegsT = VRegF32x2;
        using BRegsT = VRegF32x2;
        using CRegsT = AccRegF32x4;
        using DRegsT = AccRegF32x4;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        return typename Traits::DRegsT(
            __builtin_amdgcn_mfma_f32_16x16x16f16(*regsA, *regsB, *regsC, 0, 0, 0));
    }
};

template <>
struct amdgcn_mfma<float16_t, float16_t, 16, 16>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 16,
        };
        using ARegsT = VRegF32x2;
        using BRegsT = VRegF32x2;
        using CRegsT = AccRegF32x2;
        using DRegsT = AccRegF32x2;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        using Mfma              = amdgcn_mfma<float16_t, float32_t, 16, 16>;
        using UnpackC           = Unpack<float16_t, 2>;
        using PackD             = Pack<float16_t, 4>;
        using Convert_fp16_fp32 = amdgcn_convert<float16_t, float32_t>;
        using Convert_fp32_fp16 = amdgcn_convert<float32_t, float16_t>;

        // MFMA unit compute type is always fp32.
        // Upconvert C to fp32, do MFMA, then down convert D
        // to fp16 as 'simulated' fp16 computation
        auto Dfp32 = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(UnpackC::exec(regsC)));
        return PackD::exec(Convert_fp32_fp16::exec(Dfp32));
    }
};

template <>
struct amdgcn_mfma<float16_t, float32_t, 32, 32>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 8
        };
        using ARegsT = VRegF32x2;
        using BRegsT = VRegF32x2;
        using CRegsT = AccRegF32x16;
        using DRegsT = AccRegF32x16;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        return typename Traits::DRegsT(
            __builtin_amdgcn_mfma_f32_32x32x8f16(*regsA, *regsB, *regsC, 0, 0, 0));
    }
};

template <>
struct amdgcn_mfma<float16_t, float16_t, 32, 32>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 8,
        };
        using ARegsT = VRegF32x2;
        using BRegsT = VRegF32x2;
        using CRegsT = AccRegF32x8;
        using DRegsT = AccRegF32x8;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        using Mfma              = amdgcn_mfma<float16_t, float32_t, 32, 32>;
        using UnpackC           = Unpack<float16_t, 8>;
        using PackD             = Pack<float16_t, 16>;
        using Convert_fp16_fp32 = amdgcn_convert<float16_t, float32_t>;
        using Convert_fp32_fp16 = amdgcn_convert<float32_t, float16_t>;

        // MFMA unit compute type is always fp32.
        // Upconvert C to fp32, do MFMA, then down convert D to fp16 result;
        auto Dfp32 = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(UnpackC::exec(regsC)));
        return PackD::exec(Convert_fp32_fp16::exec(Dfp32));
    }
};

template <>
struct amdgcn_mfma<__half, float32_t, 16, 16> : public amdgcn_mfma<float16_t, float32_t, 16, 16>
{
};

template <>
struct amdgcn_mfma<__half, __half, 16, 16> : public amdgcn_mfma<float16_t, float16_t, 16, 16>
{
};

template <>
struct amdgcn_mfma<__half, float32_t, 32, 32> : public amdgcn_mfma<float16_t, float32_t, 32, 32>
{
};

template <>
struct amdgcn_mfma<__half, __half, 32, 32> : public amdgcn_mfma<float16_t, float16_t, 32, 32>
{
};

template <>
struct amdgcn_mfma<float32_t, float32_t, 16, 16>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 4
        };
        using ARegsT = VRegF32x1;
        using BRegsT = VRegF32x1;
        using CRegsT = AccRegF32x4;
        using DRegsT = AccRegF32x4;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        return typename Traits::DRegsT(
            __builtin_amdgcn_mfma_f32_16x16x4f32(*regsA, *regsB, *regsC, 0, 0, 0));
    }
};

// Single 32 x 32 block mfma
template <>
struct amdgcn_mfma<float32_t, float32_t, 32, 32>
{
    // Packed register traits
    struct Traits
    {
        enum : uint32_t
        {
            KPerMfma = 2
        };
        using ARegsT = VRegF32x1;
        using BRegsT = VRegF32x1;
        using CRegsT = AccRegF32x16;
        using DRegsT = AccRegF32x16;
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        return typename Traits::DRegsT(
            __builtin_amdgcn_mfma_f32_32x32x2f32(*regsA, *regsB, *regsC, 0, 0, 0));
    }
};

template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_mfma_MxNxK
{
    struct Traits
    {
        using MFMA = amdgcn_mfma<InputT, ComputeT, BlockM, BlockN>;

        enum : uint32_t
        {
            MfmaCount = BlockK / MFMA::Traits::KPerMfma,
            MinK      = MFMA::Traits::KPerMfma,
        };

        // Propagate individual MFMA types to full block inputs.
        using ARegsT
            = VecT<typename MFMA::Traits::ARegsT::DataT, MfmaCount * MFMA::Traits::ARegsT::size()>;
        using BRegsT
            = VecT<typename MFMA::Traits::BRegsT::DataT, MfmaCount * MFMA::Traits::BRegsT::size()>;
        using CRegsT = VecT<typename MFMA::Traits::CRegsT::DataT, MFMA::Traits::CRegsT::size()>;
        using DRegsT = VecT<typename MFMA::Traits::DRegsT::DataT, MFMA::Traits::DRegsT::size()>;

        // Sanity checks
        static_assert(BlockK >= MinK, "BlockK is not a minimum of MinK");
        static_assert(BlockK % MinK == 0, "BlockK is not a multiple of MinK");
        static_assert(std::is_same<ARegsT, BRegsT>::value,
                      "A and B registers must be of same type");
        static_assert(std::is_same<CRegsT, DRegsT>::value,
                      "C and D registers must be of same type");
        static_assert(ARegsT::size()
                          == amdgcn_io_traits<BlockM, BlockK, InputT>::PackedRegisterCount,
                      "Unexpected packed register count for A");
        static_assert(BRegsT::size()
                          == amdgcn_io_traits<BlockN, BlockK, InputT>::PackedRegisterCount,
                      "Unexpected packed register count for B");
        static_assert(CRegsT::size()
                          == amdgcn_io_traits<BlockM, BlockN, ComputeT>::PackedRegisterCount,
                      "Unexpected packed register count for C");
        static_assert(DRegsT::size()
                          == amdgcn_io_traits<BlockM, BlockN, ComputeT>::PackedRegisterCount,
                      "Unexpected packed register count for D");
    };

    __device__ static inline auto exec(typename Traits::ARegsT const& regsA,
                                       typename Traits::BRegsT const& regsB,
                                       typename Traits::CRegsT const& regsC) ->
        typename Traits::DRegsT
    {
        typename Traits::DRegsT result = regsC;

        // Accumulate into result regs
        auto aIt = regsA.template begin<Traits::MFMA::Traits::ARegsT::size()>();
        auto bIt = regsB.template begin<Traits::MFMA::Traits::BRegsT::size()>();
#pragma unroll
        for(unsigned i = 0; i < Traits::MfmaCount; i++)
        {
            result = Traits::MFMA::exec(*aIt, *bIt, result);
            aIt++;
            bIt++;
        }
        return result;
    }
};

#endif // WMMA_MFMA_H
