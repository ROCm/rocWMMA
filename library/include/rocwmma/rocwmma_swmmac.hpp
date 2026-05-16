// rocwmma_swmmac.hpp — SWMMAC backends for rocWMMA
// Layer 2 of 3: Wraps v_swmmac_i32_16x16x64_iu4 and v_swmmac_i32_16x16x32_iu8
// as rocWMMA MmaImpl backends.
//
// Follows the amdgcn_wmma pattern: provides ARegsT, BRegsT, CRegsT, DRegsT
// and exec(), enabling plug-in to rocWMMA's Mma<> driver for block iteration.
//
// Supported targets: gfx1200/gfx1201 (RDNA4) with LLVM 23+.
// On gfx11 (RDNA3), use rocWMMA's WMMA backend instead.
//
// Usage:
//   #include <rocwmma/rocwmma_swmmac.hpp>
//   // INT4
//   SwmmacInt4<>::exec(a, b, c, 0);
//   // INT8
//   SwmmacInt8<>::exec(a, b, c, 0);
//   // Sparse
//   SwmmacInt4<>::exec(a, b, c, static_cast<int32_t>(SparseSel::SPARSE));
#pragma once

#include "rocwmma_int4.hpp"
#include "internal/mma.hpp"
#include "internal/mma_traits.hpp"
#include "internal/utility/vector.hpp"

// Wave32 is mandatory for SWMMAC on gfx1200.
// Wave64 variants (_w64 suffix) exist but have different register counts.
#if ROCWMMA_HAS_SWMMAC && defined(__AMDGCN__) && !ROCWMMA_WAVE32_MODE
#error "rocwmma_swmmac.hpp requires wave32 mode (-DROCWMMA_WAVE32_MODE=1)"
#endif

#if ROCWMMA_HAS_SWMMAC

namespace rocwmma {

// ============================================================================
// SwmmacInt4 — SWMMAC INT4 backend (gfx1200/gfx1201 wave32)
//
// v_swmmac_i32_16x16x64_iu4: 16×16×64 INT4 matrix multiply-accumulate.
// Template params match the builtin's _Constant bool requirements.
// ============================================================================
template <bool ASign = true, bool BSign = true, bool CSign = true>
struct SwmmacInt4 {
    using ARegsT = SwmmacARegsT;   // <2 x i32>
    using BRegsT = SwmmacBRegsT;   // <4 x i32>
    using CRegsT = SwmmacAccumT;   // <8 x i32>
    using DRegsT = SwmmacAccumT;   // <8 x i32>

    static constexpr uint32_t BlockM = SwmmacConstants::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstants::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstants::BlockK;

    // sparse_idx: SparseSel::DENSE (0) or SparseSel::SPARSE (≠0)
    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT result;
        to_native_vector(result)
            = {__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(
                ASign, to_native_vector(a),
                BSign, to_native_vector(b),
                to_native_vector(c),
                sparse_idx, CSign)};
        return result;
    }
};

using SwmmacI4 = SwmmacInt4<true, true, true>;

// ============================================================================
// SwmmacInt8 — SWMMAC INT8 backend (gfx1200/gfx1201 wave32)
//
// v_swmmac_i32_16x16x32_iu8: 16×16×32 INT8 matrix multiply-accumulate.
// Same register layout as INT4, half the K dimension (32 vs 64).
// ============================================================================
template <bool ASign = true, bool BSign = true, bool CSign = true>
struct SwmmacInt8 {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacAccumT;
    using DRegsT = SwmmacAccumT;

    static constexpr uint32_t BlockM = SwmmacConstantsInt8::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsInt8::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsInt8::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT result;
        to_native_vector(result)
            = {__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(
                ASign, to_native_vector(a),
                BSign, to_native_vector(b),
                to_native_vector(c),
                sparse_idx, CSign)};
        return result;
    }
};

using SwmmacI8 = SwmmacInt8<true, true, true>;

// ========================================================================
// FP8 / BF8 SWMMAC backends — same A/B layout as INT4/INT8, f32 accum
//
// v_swmmac_f32_16x16x32_fp8_fp8   v_swmmac_f32_16x16x32_fp8_bf8
// v_swmmac_f32_16x16x32_bf8_fp8   v_swmmac_f32_16x16x32_bf8_bf8
//
// FP8 = E4M3 format, BF8 = E5M2 format. No sign params (FP is always signed).
// A: <2 x i32> (4 FP8/BF8 per i32), B: <4 x i32>, C/D: <8 x f32>
// ========================================================================

template <typename AType = void, typename BType = void>
struct SwmmacFp8 {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacFpAccumT;
    using DRegsT = SwmmacFpAccumT;

    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    // Specializations below provide the correct builtin
};

// FP8 × FP8 → f32
template <>
struct SwmmacFp8<float8_t, float8_t> {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacFpAccumT;
    using DRegsT = SwmmacFpAccumT;
    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

// FP8 × BF8 → f32
template <>
struct SwmmacFp8<float8_t, bfloat8_t> {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacFpAccumT;
    using DRegsT = SwmmacFpAccumT;
    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

// BF8 × FP8 → f32
template <>
struct SwmmacFp8<bfloat8_t, float8_t> {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacFpAccumT;
    using DRegsT = SwmmacFpAccumT;
    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

// BF8 × BF8 → f32
template <>
struct SwmmacFp8<bfloat8_t, bfloat8_t> {
    using ARegsT = SwmmacARegsT;
    using BRegsT = SwmmacBRegsT;
    using CRegsT = SwmmacFpAccumT;
    using DRegsT = SwmmacFpAccumT;
    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

// Convenience aliases for FP8/BF8 combos
using SwmmacFp8Fp8 = SwmmacFp8<float8_t, float8_t>;
using SwmmacFp8Bf8 = SwmmacFp8<float8_t, bfloat8_t>;
using SwmmacBf8Fp8 = SwmmacFp8<bfloat8_t, float8_t>;
using SwmmacBf8Bf8 = SwmmacFp8<bfloat8_t, bfloat8_t>;

// ========================================================================
// FP16 / BF16 SWMMAC backends (f32 accumulate)
//
// v_swmmac_f32_16x16x32_f16   v_swmmac_f32_16x16x32_bf16
//
// Different A/B register layout from INT4/INT8/FP8:
//   FP16: A=<8×f16>, B=<16×f16>
//   BF16: A=<8×i16>, B=<16×i16> (bf16 bitcast as i16)
//   C/D: <8×f32>
// ========================================================================

struct SwmmacFp16 {
    using ARegsT = SwmmacF16ARegsT;     // <8 x f16>
    using BRegsT = SwmmacF16BRegsT;     // <16 x f16>
    using CRegsT = SwmmacFpAccumT;      // <8 x f32>
    using DRegsT = SwmmacFpAccumT;

    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

struct SwmmacBf16 {
    using ARegsT = SwmmacBf16ARegsT;    // <8 x i16>
    using BRegsT = SwmmacBf16BRegsT;    // <16 x i16>
    using CRegsT = SwmmacFpAccumT;      // <8 x f32>
    using DRegsT = SwmmacFpAccumT;

    static constexpr uint32_t BlockM = SwmmacConstantsFp::BlockM;
    static constexpr uint32_t BlockN = SwmmacConstantsFp::BlockN;
    static constexpr uint32_t BlockK = SwmmacConstantsFp::BlockK;

    __device__ static inline DRegsT exec(
        ARegsT const& a, BRegsT const& b,
        CRegsT const& c, int32_t sparse_idx = 0)
    {
        DRegsT r;
        // BF16 uses native short vectors; reinterpret bf16 storage as i16
        to_native_vector(r) = {__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(
            to_native_vector(a), to_native_vector(b),
            to_native_vector(c), sparse_idx)};
        return r;
    }
};

} // namespace rocwmma

// ============================================================================
// MmaTraits — enables SwmmacInt4/Int8 as rocWMMA Mma<> driver backends
// ============================================================================
namespace rocwmma {
namespace MmaTraits_impl {

template <typename T>
struct is_swmmac : false_type {};

template <bool ASign, bool BSign, bool CSign>
struct is_swmmac<SwmmacInt4<ASign, BSign, CSign>> : true_type {};

template <bool ASign, bool BSign, bool CSign>
struct is_swmmac<SwmmacInt8<ASign, BSign, CSign>> : true_type {};

template <typename SwmmacOp>
constexpr static bool is_swmmac_v = is_swmmac<SwmmacOp>::value;

template <typename SwmmacOp>
struct swmmac_traits;

// Shared trait computation — only InputTA/TB differ between INT4/INT8
template <typename ImplT, typename TA, typename TB>
struct swmmac_traits_base {
    using Impl     = ImplT;
    using InputTA  = TA;
    using InputTB  = TB;
    using ComputeT = int32_t;

    using ARegsT = typename Impl::ARegsT;
    using BRegsT = typename Impl::BRegsT;
    using CRegsT = typename Impl::CRegsT;
    using DRegsT = typename Impl::DRegsT;

    static constexpr uint32_t BlockM = Impl::BlockM;
    static constexpr uint32_t BlockN = Impl::BlockN;
    static constexpr uint32_t BlockK = Impl::BlockK;

    static constexpr uint32_t BlockSizeA = VecTraits<ARegsT>::size();
    static constexpr uint32_t BlockSizeB = VecTraits<BRegsT>::size();
    static constexpr uint32_t BlockSizeC = VecTraits<CRegsT>::size();

    static constexpr bool is_wmma      = false;
    static constexpr bool is_mfma      = false;
    static constexpr bool is_swmmac    = true;
    static constexpr bool is_supported = true;
};

template <bool ASign, bool BSign, bool CSign>
struct swmmac_traits<SwmmacInt4<ASign, BSign, CSign>>
    : swmmac_traits_base<SwmmacInt4<ASign, BSign, CSign>, int4_t, int4_t> {};

template <bool ASign, bool BSign, bool CSign>
struct swmmac_traits<SwmmacInt8<ASign, BSign, CSign>>
    : swmmac_traits_base<SwmmacInt8<ASign, BSign, CSign>, int8_t, int8_t> {};

template <typename MmaOp>
struct MmaTraits<MmaOp, enable_if_t<is_swmmac_v<MmaOp>>>
    : public swmmac_traits<MmaOp> {};

} // namespace MmaTraits_impl
} // namespace rocwmma

#endif // ROCWMMA_HAS_SWMMAC
