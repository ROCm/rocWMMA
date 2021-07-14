#ifndef WMMA_IO_UNPACK_H
#define WMMA_IO_UNPACK_H

#include "IOTraits.h"
#include "Types.h"

template <typename DataT, uint32_t RegisterCount>
struct Unpack
{
    using BaseTraits = PackTraits<DataT>;
    struct Traits : public BaseTraits
    {
        enum : uint32_t
        {
            UnpackedRegisterCount = RegisterCount * BaseTraits::PackRatio,
            PackedRegisterCount   = RegisterCount
        };

        using InputT  = VecT<typename BaseTraits::PackedT, PackedRegisterCount>;
        using OutputT = VecT<typename BaseTraits::UnpackedT, UnpackedRegisterCount>;
    };

    // Pass-thru for no decompression
    // SFINAE on the return type
    template <typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> typename std::enable_if<
        std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value
            && (Traits::PackRatio == 1),
        decltype(std::forward<IncomingT>(input))>::type
    {
        static_assert(
            std::is_same<typename std::decay<IncomingT>::type, typename Traits::OutputT>::value,
            "Passthru requires same input and result types");
        return std::forward<IncomingT>(input);
    }

    // Actual compression needed
    template <typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> typename std::enable_if<
        std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value
            && (Traits::PackRatio > 1),
        typename Traits::OutputT&>::type
    {
        using InputT  = typename Traits::InputT;
        using OutputT = typename Traits::OutputT;

        return *reinterpret_cast<OutputT*>(&(const_cast<InputT&>(input)));
    }
};

#endif // WMMA_IO_UNPACK_H
