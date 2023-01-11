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
#ifndef ROCWMMA_ACCESSORS_HPP
#define ROCWMMA_ACCESSORS_HPP

namespace rocwmma
{
    ///
    /// DataType access
    ///

    template <typename Obj>
    struct GetDataType;

    template <typename Obj>
    using GetDataType_t = typename GetDataType<Obj>::type;

    ///
    /// IOConfig access
    ///

    template <typename Obj>
    struct GetIOConfig;

    template <typename Obj>
    using GetIOConfig_t = typename GetIOConfig<Obj>::type;

    ///
    /// IOShape access
    ///

    template <typename Obj>
    struct GetIOShape;

    template <typename Obj>
    using GetIOShape_t = typename GetIOShape<Obj>::type;

    ///
    /// IOTraits access
    ///

    template <typename Obj>
    struct GetIOTraits;

    template <typename Obj>
    using GetIOTraits_t = typename GetIOTraits<Obj>::type;

    ///
    /// MatrixLayout access
    ///

    template <typename Obj>
    struct GetMatrixLayout;

    template <typename Obj>
    using GetMatrixLayout_t = typename GetMatrixLayout<Obj>::type;

    ///
    /// DataLayout access
    ///

    template <typename Obj>
    struct GetDataLayout;

    template <typename Obj>
    using GetDataLayout_t = typename GetDataLayout<Obj>::type;

} // namespace rocwmma

#include "accessors_impl.hpp"

#endif // ROCWMMA_ACCESSORS_HPP
