#include <type_traits>

#include <Types.h>

template <uint32_t N>
using I = std::integral_constant<uint32_t, N>;

template <class... Ts, class F>
void for_each(std::tuple<Ts...>, F f)
{
    std::initializer_list<int> _ = {(f(Ts{}), 0)...}; // poor man's fold expression for C++11/14
    // (f(Ts{}), ...); // fold expression is for C++17 only
}

namespace quirks
{
    // rocBLAS does not yet support Ti/To/Tc = bf16/bf16/bf16
    template <typename InputT, typename OutputT, typename ComputeT>
    struct rocblas_supported : std::true_type
    {
    };

    template <>
    struct rocblas_supported<bfloat16_t, bfloat16_t, bfloat16_t> : std::false_type
    {
    };

    template <>
    struct rocblas_supported<int8_t, int8_t, int32_t> : std::false_type
    {
    };

    // hipcc compiler currently has bug in half-type data packing around 'v_fma_mixlo_f16'
    // where it forgot to zero out dest vgpr before writing into it
    //
    // currently the combination of column-major C/D matrix & half type output (fp16/bf16)
    // is affected
    template <typename OutputT,
              typename LayoutC,
              typename LayoutD = LayoutC,
              typename Enable  = void>
    struct hipcc_bug_half_packing : std::false_type
    {
    };

    template <typename OutputT, typename LayoutC, typename LayoutD>
    struct hipcc_bug_half_packing<
        OutputT,
        LayoutC,
        LayoutD,
        typename std::enable_if<sizeof(OutputT) == 2u && std::is_same<LayoutC, col_major>::value
                                && std::is_same<LayoutD, col_major>::value>::type> : std::true_type
    {
    };

} // namespace quirks
