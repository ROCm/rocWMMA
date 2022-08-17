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
#ifndef ROCWMMA_IO_TRAITS_HPP
#define ROCWMMA_IO_TRAITS_HPP

#include "constants.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace rocwmma
{

    namespace detail
    {
        /*
* The following class is intended to define the packing traits
* for particular datatypes. We consider that ROCWMMA uses packed
* registers. The pack ratio is how many registers resulting from
* raw IO are packed together while used in ROCWMMA.
*/

        template <typename DataT>
        struct PackTraits;

        template <>
        struct PackTraits<int8_t>
        {
            enum : uint32_t
            {
                PackRatio = 4
            };

            using UnpackedT = int8_t;
            using PackedT   = int32_t;
        };

        template <>
        struct PackTraits<uint8_t>
        {
            enum : uint32_t
            {
                PackRatio = 4
            };

            using UnpackedT = uint8_t;
            using PackedT   = uint32_t;
        };

        template <>
        struct PackTraits<uint32_t>
        {
            enum : uint32_t
            {
                PackRatio = 1
            };

            using UnpackedT = uint32_t;
            using PackedT   = uint32_t;
        };

        template <>
        struct PackTraits<int32_t>
        {
            enum : uint32_t
            {
                PackRatio = 1
            };

            using UnpackedT = int32_t;
            using PackedT   = int32_t;
        };

        template <>
        struct PackTraits<float16_t>
        {
            enum : uint32_t
            {
                PackRatio = 2 // 2 Elements combine to one
            };

            using UnpackedT = float16_t;
            using PackedT   = float32_t;
        };

        template <>
        struct PackTraits<hfloat16_t>
        {
            enum : uint32_t
            {
                PackRatio = 2 // 2 Elements combine to one
            };

            using UnpackedT = hfloat16_t;
            using PackedT   = float32_t;
        };

        template <>
        struct PackTraits<bfloat16_t>
        {
            enum : uint32_t
            {
                PackRatio = 2 // 2 Elements combine to one
            };

            using UnpackedT = bfloat16_t;
            using PackedT   = float32_t;
        };

        template <>
        struct PackTraits<float32_t>
        {
            enum : uint32_t
            {
                PackRatio = 1 // No pack
            };

            using UnpackedT = float32_t;
            using PackedT   = float32_t;
        };

        template <>
        struct PackTraits<float64_t>
        {
            enum : uint32_t
            {
                PackRatio = 1 // No pack
            };

            using UnpackedT = float64_t;
            using PackedT   = float64_t;
        };

        /*
        * The following class is intended to provide optimistic suggestions for
        * IO vector widths, up to a maximum of dwordx4 which is the largest data
        * movement instruction. Keep halving the vector width until it can fit
        * the entire block, or split evenly amongst IO iterations.
        *
        * TODO: As of ROCm 5.3, the compiler has issue with fp64 TestWidth = 2 in some
        * corner cases.
        */

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t TestWidth = std::is_same<DataT, float64_t>::value
                                           ? 8u * AMDGCN_DWORD_SIZE_BYTES / (uint32_t)sizeof(DataT)
                                           : // TODO: fp64 compiler bug
                                           4u * AMDGCN_DWORD_SIZE_BYTES / (uint32_t)sizeof(DataT)>
        struct VecWidthTraits
        {
            enum : uint32_t
            {
                ElementCount  = BlockDim * BlockK,
                ElementsPerIO = TestWidth * AMDGCN_WAVE_SIZE,
                MaxVectorWidth
                = (TestWidth <= BlockDim) && (TestWidth <= BlockK)
                          && (ElementsPerIO <= ElementCount) && (ElementCount % ElementsPerIO == 0)
                      ? TestWidth
                      : VecWidthTraits<BlockDim, BlockK, DataT, TestWidth / 2>::MaxVectorWidth
            };
        };

        template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
        struct VecWidthTraits<BlockDim, BlockK, DataT, 0>
        {
            enum : uint32_t
            {
                ElementCount   = BlockDim * BlockK,
                ElementsPerIO  = AMDGCN_WAVE_SIZE,
                MaxVectorWidth = 1
            };
        };

    } // namespace detail

    /*
* The following class provides IO meta-data that is used
* to provide static information used in inference and controlling
* certain aspects of ROCWMMA. Static asserts are also performed in this
* class to indicate specific logical issues when implementing IO
* functionality.
*/
    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth = 1>
    struct IOTraits
    {
        enum : uint32_t
        {
            // Number of threads to perform I/O operation
            ThreadsPerIO = AMDGCN_WAVE_SIZE,

            // Total number of elements in a single I/O operation
            ElementsPerIO = ThreadsPerIO * VectorWidth,

            // Number of BlockDim strides per I/O operation
            KPerIO = ceilDiv(ElementsPerIO, BlockDim),

            // Total number of elements per for the entire block
            ElementCount = BlockDim * BlockK,

            // Total number of I/O operations needed for the entire block
            IOCount = ceilDiv(ElementCount, ElementsPerIO),

            // Per-thread c++ vector storage size required for:
            // Unpacked vector = raw I/O
            // Packed vector = packed raw I/O
            UnpackedSize = ceilDiv(ElementCount, ThreadsPerIO),
            PackedSize
            = ceilDiv((uint32_t)UnpackedSize, (uint32_t)detail::PackTraits<DataT>::PackRatio),

            // Physical number of hardware vregs used to store packed data
            PackedVRegCount   = ElementCount * sizeof(DataT) / AMDGCN_REGISTER_SIZE_BYTES,
            UnpackedVRegCount = PackedVRegCount * detail::PackTraits<DataT>::PackRatio
        };

        static_assert((BlockDim <= ElementsPerIO)
                          ? ((ElementsPerIO % BlockDim) == 0 || (ElementsPerIO % BlockK) == 0)
                          : ((BlockDim % ElementsPerIO) == 0 || (BlockK % ElementsPerIO) == 0),
                      "I/O operation elements not a multiple of BlockDim");
        static_assert((ElementCount % ElementsPerIO) == 0,
                      "I/O element count not divisible into equal operations");
        static_assert((ElementCount % ThreadsPerIO) == 0,
                      "Element count must be divisible by threads per wave");
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_TRAITS_HPP
