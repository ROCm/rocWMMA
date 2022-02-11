/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_UNPACK_HPP
#define ROCWMMA_UNPACK_HPP

#include "io_traits.hpp"
#include "types.hpp"

namespace rocwmma
{

    template <typename DataT, uint32_t RegisterCount>
    struct Unpack
    {
        using BaseTraits = detail::PackTraits<DataT>;
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

} // namespace rocwmma

#endif // ROCWMMA_UNPACK_HPP
