#ifndef WMMA_MFMA_H
#define WMMA_MFMA_H

#include "Types.h"

template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_mfma;

template <>
struct amdgcn_mfma<float32_t, float32_t, 32, 32, 2>
{
    using ARegT = VRegF32x1;
    using BRegT = VRegF32x1;
    using CRegT = AccRegF32x16;
    using DRegT = AccRegF32x16;

    __device__ static inline auto exec(ARegT regA, BRegT regB, CRegT regC) -> DRegT
    {
        return DRegT(__builtin_amdgcn_mfma_f32_32x32x2f32(*regA, *regB, *regC, 0, 0, 0));
    }
};

template <>
struct amdgcn_mfma<float32_t, float32_t, 32, 64, 1>
{
    using ARegT = VRegF32x1;
    using BRegT = VRegF32x1;
    using CRegT = AccRegF32x32;
    using DRegT = AccRegF32x32;

    __device__ static inline auto exec(ARegT regA, BRegT regB, CRegT regC) -> DRegT
    {
        return DRegT(__builtin_amdgcn_mfma_f32_32x32x1f32(*regA, *regB, *regC, 1, 0, 0));
    }
};

template <>
struct amdgcn_mfma<float32_t, float32_t, 64, 32, 1>
{
    using ARegT = VRegF32x1;
    using BRegT = VRegF32x1;
    using CRegT = AccRegF32x32;
    using DRegT = AccRegF32x32;

    __device__ static inline auto exec(ARegT regA, BRegT regB, CRegT regC) -> DRegT
    {
        return DRegT(__builtin_amdgcn_mfma_f32_32x32x1f32(*regA, *regB, *regC, 0, 0, 1));
    }
};

template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_mfma_MxNxK_traits;

template <uint32_t BlockK>
struct amdgcn_mfma_MxNxK_traits<float32_t, float32_t, 32, 32, BlockK>
{
    enum : uint32_t
    {
        MfmaCount = BlockK / 2,
    };
    using MFMA     = amdgcn_mfma<float32_t, float32_t, 32, 32, 2>;
    using AInputT  = VecT<float32_t, MfmaCount>;
    using BInputT  = VecT<float32_t, MfmaCount>;
    using CInputT  = AccRegF32x16;
    using DOutputT = AccRegF32x16;
};

template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
struct amdgcn_mfma_MxNxK
{
    using Traits = amdgcn_mfma_MxNxK_traits<InputT, ComputeT, BlockM, BlockN, BlockK>;

    using MFMA   = typename Traits::MFMA;
    using ARegsT = typename Traits::AInputT;
    using BRegsT = typename Traits::BInputT;
    using CRegsT = typename Traits::CInputT;
    using DRegsT = typename Traits::DOutputT;

    static_assert(std::is_same<ARegsT, BRegsT>::value, "A and B registers must be of same type");
    static_assert(std::is_same<CRegsT, DRegsT>::value, "C and D registers must be of same type");

    __device__ static auto exec(ARegsT const& aRegs, BRegsT const& bRegs, CRegsT const& cRegs)
        -> DRegsT
    {
        DRegsT result = cRegs;
#pragma unroll
        // Accumulate into result regs
        for(unsigned i = 0; i < Traits::MfmaCount; i++)
        {
            result = MFMA::exec(aRegs[i], bRegs[i], result);
        }
        return result;
    }
};

#endif // WMMA_MFMA_H
