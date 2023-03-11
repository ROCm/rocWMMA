/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

    template <typename DataT>
    struct Unpack
    {
        using Traits = detail::PackTraits<DataT>;

        template <typename DT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto& exec(VecT<DT, VecSize> const& v)
        {
            using OutputT = VecT<typename Traits::UnpackedT, VecSize * Traits::PackRatio>;
            using InputT  = std::decay_t<decltype(v)>;

            if constexpr(Traits::PackRatio == 1u
                         || !std::is_same<DT, typename Traits::PackedT>::value)
            {
                return std::forward<InputT>(v);
            }
            else
            {
                return *reinterpret_cast<OutputT*>(&(const_cast<InputT&>(v)));
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_UNPACK_HPP
