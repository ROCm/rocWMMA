/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_TR_LOAD_HPP
#define ROCWMMA_TR_LOAD_HPP

#include "io_traits.hpp"
#include "layout.hpp"
#include "tuple.hpp"
#include "types.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_tr_load
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(VecT<DataT, VectorWidth>),
                          "Cannot vectorize input");

            using LoadT = VecT<DataT, VectorWidth>;

            ROCWMMA_UNSUPPORTED_IMPL("amdgcn_tr_load is currently only supported on gfx12+")
            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                // TODO: Implement a cross-lane ops version to perform TR shuffle
                data = *reinterpret_cast<LoadT const*>(&(dataPtr[offset]));
            }
        };

#if ROCWMMA_ARCH_GFX12

        template <>
        struct amdgcn_tr_load<float16_t, 8u>
        {
            using DataT = float16_t;
            enum : uint32_t
            {
                VectorWidth = 8u
            };

            // rocWMMA vector type
            using LoadT = VecT<DataT, VectorWidth>;

            // Intrinsic type
            using LoadIT = decltype(__builtin_amdgcn_global_load_tr_v8f16(nullptr));

            static_assert(VectorWidth > 0u, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(LoadT), "Cannot vectorize input");
            static_assert(sizeof(LoadIT) == sizeof(LoadT), "Load types must match");

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                union
                {
                    LoadIT in;
                    LoadT  out;
                } v{__builtin_amdgcn_global_load_tr_v8f16((LoadIT* const)(dataPtr + offset))};
                data = v.out;
            }
        };

        template <>
        struct amdgcn_tr_load<bfloat16_t, 8u>
        {
            using DataT = bfloat16_t;
            enum : uint32_t
            {
                VectorWidth = 8u
            };

            // rocWMMA vector type
            using LoadT = VecT<DataT, VectorWidth>;

            // Intrinsic type
            using LoadIT = decltype(__builtin_amdgcn_global_load_tr_v8i16(nullptr));

            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(LoadT), "Cannot vectorize input");
            static_assert(sizeof(LoadIT) == sizeof(LoadT), "Load types must match");

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                union
                {
                    LoadIT in;
                    LoadT  out;
                } v{__builtin_amdgcn_global_load_tr_v8i16((LoadIT* const)(dataPtr + offset))};
                data = v.out;
            }
        };

        template <>
        struct amdgcn_tr_load<int8_t, 8u>
        {
            using DataT = int8_t;
            enum : uint32_t
            {
                VectorWidth = 8u
            };

            // rocWMMA vector type
            using LoadT = VecT<DataT, VectorWidth>;

            // Intrinsic type
            using LoadIT = decltype(__builtin_amdgcn_global_load_tr4_v2i32(nullptr));

            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(LoadT), "Cannot vectorize input");
            static_assert(sizeof(LoadIT) == sizeof(LoadT), "Load types must match");

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                union
                {
                    LoadIT in;
                    LoadT  out;
                } v{__builtin_amdgcn_global_load_tr4_v2i32((LoadIT* const)(dataPtr + offset))};
                data = v.out;
            }
        };

        template <>
        struct amdgcn_tr_load<bfloat8_t, 8u>
        {
            using DataT = bfloat8_t;
            enum : uint32_t
            {
                VectorWidth = 8u
            };

            // rocWMMA vector type
            using LoadT = VecT<DataT, VectorWidth>;

            // Intrinsic type
            using LoadIT = decltype(__builtin_amdgcn_global_load_tr4_v2i32(nullptr));

            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(LoadT), "Cannot vectorize input");
            static_assert(sizeof(LoadIT) == sizeof(LoadT), "Load types must match");

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                union
                {
                    LoadIT in;
                    LoadT  out;
                } v{__builtin_amdgcn_global_load_tr4_v2i32((LoadIT* const)(dataPtr + offset))};
                data = v.out;
            }
        };

        template <>
        struct amdgcn_tr_load<float8_t, 8u>
        {
            using DataT = float8_t;
            enum : uint32_t
            {
                VectorWidth = 8u
            };

            // rocWMMA vector type
            using LoadT = VecT<DataT, VectorWidth>;

            // Intrinsic type
            using LoadIT = decltype(__builtin_amdgcn_global_load_tr4_v2i32(nullptr));

            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(LoadT), "Cannot vectorize input");
            static_assert(sizeof(LoadIT) == sizeof(LoadT), "Load types must match");

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                union
                {
                    LoadIT in;
                    LoadT  out;
                } v{__builtin_amdgcn_global_load_tr4_v2i32((LoadIT* const)(dataPtr + offset))};
                data = v.out;
            }
        };

#endif // ROCWMMA_GFX12

    } // namespace detail

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
              uint32_t VectorWidth>
    struct TRLoad
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

        struct Traits
        {
            // Raw IO on unpacked register data.
            using Loader  = detail::amdgcn_tr_load<DataT, VectorWidth>;
            using LoadT   = typename Loader::LoadT;
            using OutputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using LoadVecTraits = VecTraits<typename Traits::LoadT>;

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <std::size_t Depth = 0,
                  typename Iterator,
                  typename StrideCounts,
                  typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(Iterator&      out,
                                                       DataT const*   dataPtr,
                                                       uint32_t       ldm,
                                                       StrideCounts&& strideCounts,
                                                       Strides2d&&    strides2d)
        {
            auto strideOffset = DataLayout::fromMatrixCoord(std::get<Depth>(strides2d), ldm);
            auto strideCount  = std::get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (std::tuple_size<std::decay_t<StrideCounts>>::value - 1u))
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Loader::exec(*out, dataPtr);
                    dataPtr += strideOffset;
                    out++;
                }
            }
            // Recurse to the next nested layer
            else
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_right<Depth + 1>(out, dataPtr, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                }
            }
        }

        template <std::size_t Depth = 0,
                  typename Iterator,
                  typename StrideCounts,
                  typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_left(Iterator&      out,
                                                      DataT const*   dataPtr,
                                                      uint32_t       ldm,
                                                      StrideCounts&& strideCounts,
                                                      Strides2d&&    strides2d)
        {
            constexpr auto size = std::tuple_size<std::decay_t<StrideCounts>>::value;

            auto strideOffset
                = DataLayout::fromMatrixCoord(std::get<size - 1 - Depth>(strides2d), ldm);
            auto strideCount = std::get<size - 1 - Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (size - 1u))
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Loader::exec(*out, dataPtr);
                    dataPtr += strideOffset;
                    out++;
                }
            }
            // Recurse to the next nested layer
            else
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_left<Depth + 1>(out, dataPtr, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                }
            }
        }

        ROCWMMA_DEVICE static void
            exec(typename Traits::OutputT& data, DataT const* dataPtr, uint32_t ldm)
        {
            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset2d = MatrixLayout::baseOffset();
            auto it           = makeVectorIterator<LoadVecTraits::size()>(data).begin();

            static_assert(decltype(it)::range() == IOTraits::IOCount,
                          "IOCount inconsistent with iterator range");

            // Make sure that the IOCount is consistent with the number of total strides
            static_assert(IOTraits::IOCount
                              == std::apply([](auto... items) { return (items * ...); },
                                            MatrixLayout::strideCounts()),
                          "IOCount inconsistent with total strides");

            // Unroll loading in each strided dimension
            unroll_right(it,
                         dataPtr + DataLayout::fromMatrixCoord(baseOffset2d, ldm),
                         ldm,
                         MatrixLayout::strideCounts(),
                         MatrixLayout::strides());
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_TR_LOAD_HPP
