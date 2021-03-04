#ifndef WMMA_IO_UNPACK_H
#define WMMA_IO_UNPACK_H

template<typename DataT>
struct Unpack;

template<>
struct Unpack<float16_t>
{
    struct Traits
    {
        enum : uint32_t
        {
            UnpackRatio = 2 // 1 element splits into 2 
        };

        using UnpackedT = float16_t;
        using PackedT = float32_t;

        using InputT = VecT<PackedT, 1>;
        using OutputT = VecT<UnpackedT, UnpackRatio>;
    };

    __device__ static inline auto exec(typename Traits::InputT const& input) -> typename Traits::OutputT
    {
        using OutputT = typename Traits::OutputT;

        return OutputT(*reinterpret_cast<typename OutputT::VecType*>(
            &(const_cast<typename Traits::InputT&>(input).v())));
    }
};

template<>
struct Unpack<float32_t>
{
    struct Traits
    {
        enum : uint32_t
        {
            UnpackRatio = 1 // No unpack
        };

        using UnpackedT = float32_t;
        using PackedT = float32_t;

        using InputT = VecT<PackedT, 1>;
        using OutputT = VecT<UnpackedT, UnpackRatio>;
    };

    // Passthrough
    template<typename IncomingT>
    __device__ static inline auto exec(IncomingT&& input) -> decltype(std::forward<IncomingT>(input))
    {
        return std::forward<IncomingT>(input);
    }
};

template<typename DataT, uint32_t RegisterCount>
struct UnpackRegs
{
    struct Traits : public Unpack<DataT>::Traits
    {
        enum : uint32_t
        {
            UnpackedRegisterCount = RegisterCount * Unpack<DataT>::Traits::UnpackRatio,
            PackedRegisterCount = RegisterCount
        };

        using Unpacker = Unpack<DataT>;
        using InputT = VecT<typename Unpacker::Traits::PackedT, PackedRegisterCount>;
        using OutputT = VecT<typename Unpacker::Traits::UnpackedT, UnpackedRegisterCount>;
    };

    // Pass-thru for no decompression
    // SFINAE on the return type
    template<typename IncomingT>
    __device__ static inline
    typename std::enable_if<std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value && (Traits::UnpackRatio == 1), typename Traits::OutputT>::type
    exec(IncomingT&& input)
    {
        static_assert(std::is_same<typename std::decay<IncomingT>::type, typename Traits::OutputT>::value, "Passthru requires same input and result types");
        return std::forward<IncomingT>(input);
    }

    // Actual compression needed
    template<typename IncomingT>
    __device__ static inline
    typename std::enable_if<std::is_same<typename std::decay<IncomingT>::type, typename Traits::InputT>::value && (Traits::UnpackRatio > 1), typename Traits::OutputT>::type
    exec(IncomingT&& input)
    {
        using Unpacker = typename Traits::Unpacker;
        using OutputT = typename Traits::OutputT;

        OutputT result;
        auto it = result.template begin<Traits::UnpackRatio>();

        static_assert(decltype(it)::Range == Traits::PackedRegisterCount, "Iterator range doesn't match packed register count");

#pragma unroll
        for(int i = 0; i < Traits::PackedRegisterCount; ++i)
        {
            *it = *Unpacker::exec(input[i]);
            it++;
        }
        return result;
    }
};

#endif // WMMA_IO_UNPACK_H
