#ifndef _IO_PACK_H
#define _IO_PACK_H

#include "Types.h"
#include "Utils.h"

template<typename DataT>
struct Pack;

template<>
struct Pack<float16_t>
{
    struct Traits
    {
        enum : uint32_t
        {
            PackRatio = 2 // 2 Elements combine to one
        };

        using UnpackedT = float16_t;
        using PackedT = float32_t;

        using InputT = VecT<UnpackedT, PackRatio>;
        using OutputT = VecT<PackedT, 1>;
    };
    
    __device__ static inline auto exec(typename Traits::InputT const& input) -> typename Traits::OutputT
    {  
        using OutputT = typename Traits::OutputT;

        return OutputT(*reinterpret_cast<typename OutputT::VecType*>(
            &(const_cast<typename Traits::InputT&>(input).v())));
    }
};

template<>
struct Pack<float32_t>
{
    struct Traits
    {
        enum : uint32_t
        {
            PackRatio = 1 // No pack
        };

        using UnpackedT = float32_t;
        using PackedT = float32_t;

        using InputT = VecT<UnpackedT, PackRatio>;
        using OutputT = VecT<PackedT, 1>;
    };
    
    // Passthrough
    template<typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> decltype(std::forward<IncomingT>(input))
    {
        return std::forward<IncomingT>(input);
    }
};

template<typename DataT, uint32_t RegisterCount>
struct PackRegs
{
    struct Traits : public Pack<DataT>::Traits
    {
        enum : uint32_t
        {
            UnpackedRegisterCount = RegisterCount,
            PackedRegisterCount = RegisterCount / Pack<DataT>::Traits::PackRatio
        };

        static_assert(RegisterCount % Pack<DataT>::Traits::PackRatio == 0, "RegisterCount must be divisible by PackRatio");

        using Packer = Pack<DataT>;
        using InputT = VecT<typename Packer::Traits::UnpackedT, UnpackedRegisterCount>;
        using OutputT = VecT<typename Packer::Traits::PackedT, PackedRegisterCount>;
    };

    // Pass-thru for no compression
    // SFINAE on the return type
    template<typename IncomingT>
    __device__ static inline
    typename std::enable_if<std::is_same<IncomingT, typename Traits::InputT>::value && (Traits::PackRatio == 1), typename Traits::OutputT>::type
    exec(IncomingT&& input)
    {
        static_assert(std::is_same<IncomingT, typename Traits::OutputT>::value, "Passthru requires same input and result types");
        return std::forward<IncomingT>(input);
    }

    // Actual compression needed
    template<typename IncomingT>
    __device__ static inline
    typename std::enable_if<std::is_same<IncomingT, typename Traits::InputT>::value && (Traits::PackRatio > 1), typename Traits::OutputT>::type
    exec(IncomingT&& input)
    {
        using Packer = typename Traits::Packer;
        using OutputT = typename Traits::OutputT;

        OutputT result;
        auto it = input.template begin<Traits::PackRatio>();

        static_assert(decltype(it)::Range == Traits::PackedRegisterCount, "Iterator range doesn't match packed register count");

#pragma unroll
        for(int i = 0; i < Traits::PackedRegisterCount; ++i)
        {
            result[i] = *Packer::exec(*it);
            it++;
        }
        return result;
    }
};

#endif // _IO_PACK_H