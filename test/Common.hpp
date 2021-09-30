/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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
