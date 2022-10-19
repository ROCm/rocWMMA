#ifndef ROCWMMA_VECTOR_HPP
#define ROCWMMA_VECTOR_HPP

#include "types.hpp"
#include "types_ext.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}
namespace rocwmma
{
    namespace detail
    {
        namespace ArithmeticOp
        {
            struct Add
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs + rhs;
                }
            };
            struct Sub
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs - rhs;
                }
            };
            struct Mult
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs * rhs;
                }
            };

            struct Div
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs / rhs;
                }
            };

            struct Mod
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs % rhs;
                }
            };

            struct Minus
            {
                template <typename TT,
                          typename std::enable_if<std::is_signed<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return -lhs;
                }
            };

        } // namespace ArithmeticOp

        namespace BitwiseOp
        {
            struct And
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs & rhs;
                }
            };

            struct Or
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs | rhs;
                }
            };

            struct Not
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return ~lhs;
                }
            };

            struct Xor
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs ^ rhs;
                }
            };

            struct ShiftR
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

            struct ShiftL
            {
                template <typename TT,
                          typename std::enable_if<std::is_integral<TT>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

        } // namespace BitwiseOp

        namespace LogicalOp
        {
            struct And
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs && rhs;
                }
            };

            struct Or
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs || rhs;
                }
            };

            struct Not
            {
                template <typename TT,
                          typename std::enable_if<std::is_convertible<TT, bool>{}>::type* = nullptr>
                __HOST_DEVICE__ constexpr static inline auto exec(TT lhs)
                {
                    return !lhs;
                }
            };

        } // namespace LogicalOp

        namespace RelationalOp
        {
            struct Eq
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs == rhs;
                }
            };

            struct Neq
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs != rhs;
                }
            };

            struct Gte
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs >= rhs;
                }
            };

            struct Lte
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs <= rhs;
                }
            };

            struct Gt
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs > rhs;
                }
            };

            struct Lt
            {
                template <typename TT>
                __HOST_DEVICE__ constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs < rhs;
                }
            };

        } // namespace RelationalOp

    } // namespace detail

    template <typename T, unsigned int Rank>
    struct non_native_vector_base
    {
        /// Types
        // Helpers for fold expansion
        template <uint32_t... ns>
        using SeqT = std::integer_sequence<uint32_t, ns...>;
        using Seq  = std::make_integer_sequence<uint32_t, Rank>;

        // Relational vector result type
        using BoolVecT = uint32_t __attribute__((ext_vector_type(next_pow2(Rank))));
        using VecT     = non_native_vector_base<T, Rank>;

        /// Ctor, dtor, assignment
        __HOST_DEVICE__
        non_native_vector_base() = default;

        __HOST_DEVICE__
        explicit constexpr non_native_vector_base(T x_) noexcept
        {
            (*this) = bCast(x_, Seq{});
        }

        template <typename... Ts,
                  typename U                                                        = T,
                  typename std::enable_if<(sizeof...(Ts) > 1) && (sizeof...(Ts) == Rank)
                                          && (std::is_same<U, Ts>{} && ...)>::type* = nullptr>
        __HOST_DEVICE__ constexpr non_native_vector_base(Ts... args) noexcept
            : d{args...}
        {
        }

        __HOST_DEVICE__
        constexpr non_native_vector_base(const VecT&) = default;

        __HOST_DEVICE__
        constexpr non_native_vector_base(VecT&&) = default;

        __HOST_DEVICE__
        ~non_native_vector_base() = default;

        __HOST_DEVICE__
        VecT& operator=(const VecT&) = default;

        __HOST_DEVICE__
        VecT& operator=(VecT&&) = default;

        __HOST_DEVICE__
        T& operator[](unsigned int idx) noexcept
        {
            return d[idx];
        }

        __HOST_DEVICE__
        T operator[](unsigned int idx) const noexcept
        {
            return d[idx];
        }

        template <class BinOp, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline auto
            binOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return VecT{(BinOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        template <class UnOp, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline auto unOp(VecT const& lhs,
                                                          SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return VecT{(UnOp::exec(lhs.d[indices]))...};
        }

        template <class BoolOp, uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline auto
            boolOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return BoolVecT{(BoolOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        template <uint32_t... indices>
        __HOST_DEVICE__ constexpr static inline auto bCast(T val, SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return VecT{(val + static_cast<T>(indices & 0u))...};
        }

        __HOST_DEVICE__
        inline VecT& operator+=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::ArithmeticOp::Add>(*this, x_, Seq{}));
        }

        __HOST_DEVICE__
        VecT& operator-=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::ArithmeticOp::Sub>(*this, x_, Seq{}));
        }

        __HOST_DEVICE__
        VecT& operator*=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::ArithmeticOp::Mult>(*this, x_, Seq{}));
        }

        __HOST_DEVICE__
        VecT& operator/=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::ArithmeticOp::Div>(*this, x_, Seq{}));
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator%=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::ArithmeticOp::Mod>(*this, x_, Seq{}));
        }

        template <typename U = T, typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT operator-() const noexcept
        {
            return unOp<detail::ArithmeticOp::Minus>(*this, Seq{});
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator&=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::BitwiseOp::And>(*this, x_, Seq{}));
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator|=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::BitwiseOp::Or>(*this, x_, Seq{}));
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT operator~() const noexcept
        {
            return unOp<detail::BitwiseOp::Not>(*this, Seq{});
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator^=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::BitwiseOp::Xor>(*this, x_, Seq{}));
        }

        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator>>=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::BitwiseOp::ShiftR>(*this, x_, Seq{}));
        }
        template <typename U = T, typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__ VecT& operator<<=(const VecT& x_) noexcept
        {
            return (*this = binOp<detail::BitwiseOp::ShiftL>(*this, x_, Seq{}));
        }

        __HOST_DEVICE__
        auto operator==(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Eq>(*this, x_, Seq{});
        }

        __HOST_DEVICE__
        auto operator!=(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Neq>(*this, x_, Seq{});
        }

        __HOST_DEVICE__
        auto operator>=(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Gte>(*this, x_, Seq{});
        }

        __HOST_DEVICE__
        auto operator<=(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Lte>(*this, x_, Seq{});
        }

        __HOST_DEVICE__
        auto operator>(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Gt>(*this, x_, Seq{});
        }

        __HOST_DEVICE__
        auto operator<(const VecT& x_) const noexcept
        {
            return boolOp<detail::RelationalOp::Lt>(*this, x_, Seq{});
        }

        /// Storage
        T d[Rank];
    };

} // namespace rocwmma

#if __HIP_CLANG_ONLY__
#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK1(TYPE) \
    struct                                      \
    {                                           \
        TYPE x;                                 \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK2(TYPE) \
    struct                                      \
    {                                           \
        TYPE x;                                 \
        TYPE y;                                 \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK3(TYPE) \
    struct                                      \
    {                                           \
        TYPE x;                                 \
        TYPE y;                                 \
        TYPE z;                                 \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK4(TYPE) \
    struct                                      \
    {                                           \
        TYPE x;                                 \
        TYPE y;                                 \
        TYPE z;                                 \
        TYPE w;                                 \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK8(TYPE) \
    struct                                      \
    {                                           \
        TYPE x0;                                \
        TYPE y0;                                \
        TYPE z0;                                \
        TYPE w0;                                \
        TYPE x1;                                \
        TYPE y1;                                \
        TYPE z1;                                \
        TYPE w1;                                \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK16(TYPE) \
    struct                                       \
    {                                            \
        TYPE x0;                                 \
        TYPE y0;                                 \
        TYPE z0;                                 \
        TYPE w0;                                 \
        TYPE x1;                                 \
        TYPE y1;                                 \
        TYPE z1;                                 \
        TYPE w1;                                 \
        TYPE x2;                                 \
        TYPE y2;                                 \
        TYPE z2;                                 \
        TYPE w2;                                 \
        TYPE x3;                                 \
        TYPE y3;                                 \
        TYPE z3;                                 \
        TYPE w3;                                 \
    };

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK32(TYPE) \
    struct                                       \
    {                                            \
        TYPE x0;                                 \
        TYPE y0;                                 \
        TYPE z0;                                 \
        TYPE w0;                                 \
        TYPE x1;                                 \
        TYPE y1;                                 \
        TYPE z1;                                 \
        TYPE w1;                                 \
        TYPE x2;                                 \
        TYPE y2;                                 \
        TYPE z2;                                 \
        TYPE w2;                                 \
        TYPE x3;                                 \
        TYPE y3;                                 \
        TYPE z3;                                 \
        TYPE w3;                                 \
        TYPE x4;                                 \
        TYPE y4;                                 \
        TYPE z4;                                 \
        TYPE w4;                                 \
        TYPE x5;                                 \
        TYPE y5;                                 \
        TYPE z5;                                 \
        TYPE w5;                                 \
        TYPE x6;                                 \
        TYPE y6;                                 \
        TYPE z6;                                 \
        TYPE w6;                                 \
        TYPE x7;                                 \
        TYPE y7;                                 \
        TYPE z7;                                 \
        TYPE w7;                                 \
    };

#else

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK1(TYPE) hip_impl::Scalar_accessor<TYPE, Native_vec_, 0> x;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK2(TYPE) \
    ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK1(TYPE)     \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 1> y;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK3(TYPE) \
    ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK2(TYPE)     \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 2> z;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK4(TYPE) \
    ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK3(TYPE)     \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 3> w;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK8(TYPE)         \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 0> x0; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 1> y0; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 2> z0; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 3> w0; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 4> x1; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 5> y1; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 6> z1; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 7> w1;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK16(TYPE)         \
    ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK8(TYPE)              \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 8>  x2; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 9>  y2; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 10> z2; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 11> w2; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 12> x3; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 13> y3; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 14> z3; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 15> w3;

#define ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK32(TYPE)         \
    ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK16(TYPE)             \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 16> x4; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 17> y4; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 18> z4; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 19> w4; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 20> x5; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 21> y5; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 22> z5; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 23> w5; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 24> x6; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 25> y6; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 26> z6; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 27> w6; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 28> x7; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 29> y7; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 30> z7; \
    hip_impl::Scalar_accessor<TYPE, Native_vec_, 31> w7;

#endif

#define ROCWMMA_NON_NATIVE_VECTOR_STORAGE_IMPL(TYPE, RANK)           \
    using Native_vec_ = rocwmma::non_native_vector_base<TYPE, RANK>; \
                                                                     \
    union alignas(hip_impl::next_pot(RANK * sizeof(TYPE)))           \
    {                                                                \
        Native_vec_ data;                                            \
        ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK##RANK(TYPE);                \
    };

#define ROCWMMA_NATIVE_VECTOR_STORAGE_IMPL(TYPE, RANK)               \
    using Native_vec_ = TYPE __attribute__((ext_vector_type(RANK))); \
                                                                     \
    union                                                            \
    {                                                                \
        Native_vec_ data;                                            \
        ROCWMMA_ACCESSOR_ALIAS_IMPL_RANK##RANK(TYPE);                \
    };

#define ROCWMMA_REGISTER_VECTOR_BASE(TYPE, RANK, STORAGE_IMPL)                    \
    template <>                                                                   \
    struct HIP_vector_base<TYPE, RANK>                                            \
    {                                                                             \
        STORAGE_IMPL(TYPE, RANK);                                                 \
                                                                                  \
        using value_type = TYPE;                                                  \
                                                                                  \
        __HOST_DEVICE__                                                           \
        HIP_vector_base() = default;                                              \
                                                                                  \
        template <typename... Args>                                               \
        __HOST_DEVICE__ constexpr HIP_vector_base(Args... args) noexcept          \
            : data{args...}                                                       \
        {                                                                         \
        }                                                                         \
                                                                                  \
        __HOST_DEVICE__                                                           \
        constexpr HIP_vector_base(const HIP_vector_base&) = default;              \
                                                                                  \
        __HOST_DEVICE__                                                           \
        constexpr HIP_vector_base(HIP_vector_base&&) = default;                   \
                                                                                  \
        __HOST_DEVICE__                                                           \
        ~HIP_vector_base() = default;                                             \
                                                                                  \
        __HOST_DEVICE__                                                           \
        HIP_vector_base& operator=(const HIP_vector_base& x_) noexcept = default; \
    };

#define ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(TYPE, RANK) \
    ROCWMMA_REGISTER_VECTOR_BASE(TYPE, RANK, ROCWMMA_NATIVE_VECTOR_STORAGE_IMPL)

#define ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(TYPE, RANK) \
    ROCWMMA_REGISTER_VECTOR_BASE(TYPE, RANK, ROCWMMA_NON_NATIVE_VECTOR_STORAGE_IMPL)

#define ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(RANK)                \
    template <>                                                           \
    __HOST_DEVICE__ inline HIP_vector_type<rocwmma::bfloat16_t, RANK>&    \
        HIP_vector_type<rocwmma::bfloat16_t, RANK>::operator++() noexcept \
    {                                                                     \
        return *this += HIP_vector_type<rocwmma::bfloat16_t, RANK>{       \
                   static_cast<rocwmma::bfloat16_t>(1.0f)};               \
    }                                                                     \
                                                                          \
    template <>                                                           \
    __HOST_DEVICE__ inline HIP_vector_type<rocwmma::bfloat16_t, RANK>&    \
        HIP_vector_type<rocwmma::bfloat16_t, RANK>::operator--() noexcept \
    {                                                                     \
        return *this -= HIP_vector_type<rocwmma::bfloat16_t, RANK>{       \
                   static_cast<rocwmma::bfloat16_t>(1.0f)};               \
    }

/// Register bfloat16 vector types
// At this point, HIP_vector_type should already be defined (via including hip_vector_types).
// Entry point for bfloat16_t is by defining the HIP_vector_base<blfloat16_t, *> backends:
// bfloat16_t cannot be vectorized natively.

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 1);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 2);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 3);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 4);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::bfloat16_t, 32);

// Quirk: explicit specialization for ++ / -- operators in HIP_vector_type<bfloat16_t, *>.
// Why? bfloat16_t doesn't have automatic conversion from integers;
// Need to override such that in(de)crements use 1.f instead of 1(int)

ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(1);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(2);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(3);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(4);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(8);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(16);
ROCWMMA_SPECIALIZE_BFLOAT16_VECTOR_OPERATORS(32);

// Explicit instantiation for vector sizes up to 32
template struct HIP_vector_type<rocwmma::bfloat16_t, 1>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 2>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 3>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 4>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 8>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 16>;
template struct HIP_vector_type<rocwmma::bfloat16_t, 32>;

/// Register __half vector types

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 1);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 2);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 3);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 4);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::hfloat16_t, 32);

// Explicit instantiation for vector sizes up to 32
template struct HIP_vector_type<rocwmma::hfloat16_t, 1>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 2>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 3>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 4>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 8>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 16>;
template struct HIP_vector_type<rocwmma::hfloat16_t, 32>;

#if __has_attribute(ext_vector_type)

ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float16_t, 8);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float16_t, 16);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float16_t, 32);

ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float32_t, 8);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float32_t, 16);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float32_t, 32);

ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float64_t, 8);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float64_t, 16);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::float64_t, 32);

ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int8_t, 8);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int8_t, 16);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int8_t, 32);

ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int32_t, 8);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int32_t, 16);
ROCWMMA_REGISTER_NATIVE_VECTOR_BASE(rocwmma::int32_t, 32);

#else

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float16_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float16_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float16_t, 32);

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float32_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float32_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float32_t, 32);

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float64_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float64_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::float64_t, 32);

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int8_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int8_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int8_t, 32);

ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int32_t, 8);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int32_t, 16);
ROCWMMA_REGISTER_NON_NATIVE_VECTOR_BASE(rocwmma::int32_t, 32);

#endif

// template struct HIP_vector_type<rocwmma::float16_t, 1>;
// template struct HIP_vector_type<rocwmma::float16_t, 2>;
// template struct HIP_vector_type<rocwmma::float16_t, 3>;
// template struct HIP_vector_type<rocwmma::float16_t, 4>;
// template struct HIP_vector_type<rocwmma::float16_t, 8>;
// template struct HIP_vector_type<rocwmma::float16_t, 16>;
// template struct HIP_vector_type<rocwmma::float16_t, 32>;

// template struct HIP_vector_type<rocwmma::float32_t, 1>;
// template struct HIP_vector_type<rocwmma::float32_t, 2>;
// template struct HIP_vector_type<rocwmma::float32_t, 3>;
// template struct HIP_vector_type<rocwmma::float32_t, 4>;
// template struct HIP_vector_type<rocwmma::float32_t, 8>;
// template struct HIP_vector_type<rocwmma::float32_t, 16>;
// template struct HIP_vector_type<rocwmma::float32_t, 32>;

// template struct HIP_vector_type<rocwmma::float64_t, 1>;
// template struct HIP_vector_type<rocwmma::float64_t, 2>;
// template struct HIP_vector_type<rocwmma::float64_t, 3>;
// template struct HIP_vector_type<rocwmma::float64_t, 4>;
// template struct HIP_vector_type<rocwmma::float64_t, 8>;
// template struct HIP_vector_type<rocwmma::float64_t, 16>;
// template struct HIP_vector_type<rocwmma::float64_t, 32>;

// template struct HIP_vector_type<rocwmma::int8_t, 1>;
// template struct HIP_vector_type<rocwmma::int8_t, 2>;
// template struct HIP_vector_type<rocwmma::int8_t, 3>;
// template struct HIP_vector_type<rocwmma::int8_t, 4>;
// template struct HIP_vector_type<rocwmma::int8_t, 8>;
// template struct HIP_vector_type<rocwmma::int8_t, 16>;
// template struct HIP_vector_type<rocwmma::int8_t, 32>;

// template struct HIP_vector_type<rocwmma::int32_t, 1>;
// template struct HIP_vector_type<rocwmma::int32_t, 2>;
// template struct HIP_vector_type<rocwmma::int32_t, 3>;
// template struct HIP_vector_type<rocwmma::int32_t, 4>;
// template struct HIP_vector_type<rocwmma::int32_t, 8>;
// template struct HIP_vector_type<rocwmma::int32_t, 16>;
// template struct HIP_vector_type<rocwmma::int32_t, 32>;

#endif // ROCWMMA_VECTOR_HPP
