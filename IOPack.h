#ifndef _IO_PACK_H
#define _IO_PACK_H

#include "IOTraits.h"
#include "Types.h"

template <typename DataT, uint32_t RegisterCount>
struct Pack
{
    using BaseTraits = PackTraits<DataT>;
    struct Traits : public BaseTraits
    {
        enum : uint32_t
        {
            UnpackedRegisterCount = RegisterCount,
            PackedRegisterCount   = RegisterCount / BaseTraits::PackRatio
        };

        static_assert(RegisterCount % BaseTraits::PackRatio == 0,
                      "RegisterCount must be divisible by PackRatio");

        using InputT  = VecT<typename BaseTraits::UnpackedT, UnpackedRegisterCount>;
        using OutputT = VecT<typename BaseTraits::PackedT, PackedRegisterCount>;
    };

    // Pass-thru for no compression
    // SFINAE on the return type
    template <typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> typename std::enable_if<
        std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value
            && (Traits::PackRatio == 1),
        decltype(std::forward<IncomingT>(input))>::type
    {
        static_assert(std::is_same<typename Traits::InputT, typename Traits::OutputT>::value,
                      "Passthru requires same input and result types");
        return std::forward<IncomingT>(input);
    }

    // Actual compression needed
    template <typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> typename std::enable_if<
        std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value
            && (Traits::PackRatio > 1),
        typename Traits::OutputT>::type

    {
        using InputT  = typename Traits::InputT;
        using OutputT = typename Traits::OutputT;

        return OutputT(
            *reinterpret_cast<typename OutputT::VecType*>(&(const_cast<InputT&>(input).v())));
    }
};

#endif // _IO_PACK_H
