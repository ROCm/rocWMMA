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
#ifndef ROCWMMA_TYPES_HPP
#define ROCWMMA_TYPES_HPP

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <array>
#include <type_traits>
#include <utility>

namespace rocwmma
{

    /**
 * \ingroup rocwmma
 * \defgroup DataTypes
 *
 * @brief Definition and metadata on supported data types of matrices.
 *
 * Native Data Types:
 * float64_t = f64 = double
 * float = f32
 * _Float16 = f16
 * int8
 * uint8
 * int16
 * int32
 * uint32
 *
 *
 * Non-Native Data Types:
 * h16 = __half
 * bf16 = bfloat16
 *
 */

    // Native types
    using float16_t = _Float16;
    using float32_t = float;
    using float64_t = double;
    using int8_t    = signed char;
    using uint8_t   = unsigned char;
    using int16_t   = short;
    using int32_t   = int;
    using uint32_t  = unsigned int;
    using index_t   = int32_t;

    // Non-native types
    using bfloat16_t = hip_bfloat16;
    using hfloat16_t = __half;

    // clang-format off

    // Data layout meta-tags
    /*! \struct row_major
 *  \brief Data/In-memory Layout as Row Major
 */
    struct row_major{};
    /*! \struct col_major
 *  \brief Data/In-memory Layout as Column Major
 */
    struct col_major{};

    // Fragment usage meta-tags
    /*! \struct matrix_a
 *  \brief Input Matrix A
 */
    struct matrix_a{};
    /*! \struct matrix_b
 *  \brief Input Matrix B
 */
    struct matrix_b{};
    /*! \struct accumulator
 *  \brief Input/Output Matrix Accumulator
 */
    struct accumulator{};

    // clang-format on

    /*! \struct layout_t
 *  \brief Definition of Runtime data layout flags
 *  @var mem_row_major
 *  @var mem_col_major
 */
    enum layout_t : uint32_t
    {
        mem_row_major,
        mem_col_major
    };

    namespace detail
    {
        using Coord2d = std::pair<uint32_t, uint32_t>;

        /*! \struct VectorStorage
        *  \brief Vectorized internal storage
        *  @tparam T Storage type
        *  @tparam Elements No of Elements in the vector
        *  @tparam IsNativType Native or rocWMMA defined
        */
        template <typename T,
                  int Elements,
                  typename IsNativeType = typename std::is_fundamental<T>::type>
        struct VectorStorage;

        // Native types can use explicit vector extension
        template <typename T, int Elements>
        struct VectorStorage<T, Elements, std::true_type>
        {
            using Type = T __attribute__((ext_vector_type(Elements)));
        };

        // Non-native types can use std::arrays.
        // std::arrays has the same memory footprint as C-arrays
        // but carry extra useful functionality.
        // This allows us to use non-native data types as "vectors".
        template <typename T, int Elements>
        struct VectorStorage<T, Elements, std::false_type>
        {
            using Type = std::array<T, Elements>;
        };

        // Finally, internal vector storage
        // shall only vectorize for elements > 1
        template <typename T, int Elements>
        using VectorStorage_internal = typename std::
            conditional<Elements == 1, T, typename VectorStorage<T, Elements>::Type>::type;

    } // namespace detail

    /*! \class VecT
    *  \brief  Functional vector class
    *  @tparam T Vector data type
    *  @tparam VecSize Vector storage size
    */
    template <typename T, uint32_t VecSize>
    class __align__(4) VecT
    {
    public: // Types
        using StorageT                 = detail::VectorStorage_internal<T, VecSize>;
        using DataT                    = T;
        constexpr static uint32_t Size = VecSize;

    private: // Actual vector storage
        union
        {
            StorageT v; // Vectorized representation
            DataT    e[VecSize]; // Element array representation

            static_assert(sizeof(StorageT) == sizeof(DataT[VecSize]),
                          "Unable to vectorize with StorageT");
        };

    private:
        // Vector iterator class: handles for const and non-const vectors
        template <uint32_t SubVecSize, bool IsConst>
        class Iterator
        {
        private:
            using ParentT = typename std::
                conditional_t<IsConst, VecT<DataT, VecSize> const, VecT<DataT, VecSize>>;

            int32_t  mIndex = 0;
            ParentT& mParent;

        public:
            struct Traits
            {
                // Iterates over sub-vector type (may be > 1)
                using ItVecT = typename std::
                    conditional_t<IsConst, VecT<DataT, SubVecSize> const, VecT<DataT, SubVecSize>>;

                enum : int32_t
                {
                    Range = VecSize / SubVecSize
                };
            };

            static_assert(VecSize % SubVecSize == 0, "VecSize not iterable by ItVecSize");

            __device__ constexpr Iterator(ParentT& parent, uint32_t startIndex = 0);
            __device__ ~Iterator() = default;

            __device__ inline typename Traits::ItVecT&       operator*() const;
            __device__ inline typename Traits::ItVecT&       operator*();
            __device__ inline Iterator<SubVecSize, IsConst>& operator++();
            __device__ inline Iterator<SubVecSize, IsConst>& operator++(int);
            __device__ inline Iterator<SubVecSize, IsConst>& operator+=(int i);
            __device__ inline Iterator<SubVecSize, IsConst>& operator--();
            __device__ inline Iterator<SubVecSize, IsConst>& operator--(int);
            __device__ inline Iterator<SubVecSize, IsConst>& operator-=(int i);

            __device__ inline Iterator<SubVecSize, IsConst> next() const;
            __device__ inline Iterator<SubVecSize, IsConst> prev() const;
            __device__ inline int32_t                       index() const;
            __device__ bool                                 valid() const;

            __device__ constexpr static inline int32_t range();
            __device__ constexpr static inline bool    isConst();
        };

    public: // Vector iterator aliases
        template <uint32_t SubVecSize>
        using iterator = Iterator<SubVecSize, false>;

        template <uint32_t SubVecSize>
        using const_iterator = Iterator<SubVecSize, true>;

    public:
        // Ctor / dtor / copy
        __device__ VecT() = default;
        __device__ inline VecT(VecT const& other);
        __device__ inline VecT(StorageT const& other);
        __device__ VecT(StorageT && other);
        __device__ ~VecT() = default;

        // Accessors
        __device__ inline DataT&          operator[](uint32_t index);
        __device__ inline DataT const&    operator[](uint32_t index) const;
        __device__ inline StorageT&       operator*();
        __device__ inline StorageT const& operator*() const;

        __device__ constexpr static inline uint32_t size();

        // mutable iterators
        template <uint32_t SubVecSize = 1>
        __device__ inline iterator<SubVecSize> begin();
        template <uint32_t SubVecSize = 1>
        __device__ inline iterator<SubVecSize> end();
        template <uint32_t SubVecSize = 1>
        __device__ inline iterator<SubVecSize> it(uint32_t startIndex = 0);

        // const iterators
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> end() const;
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> begin() const;
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> it(uint32_t startIndex = 0) const;
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> cend() const;
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> cbegin() const;
        template <uint32_t SubVecSize = 1>
        __device__ inline const_iterator<SubVecSize> cit(uint32_t startIndex = 0) const;
    };

    // MFMA vector registers
    using VRegI8x1  = VecT<int8_t, 1>; // Single i8 register
    using VRegI8x2  = VecT<int8_t, 2>; // Two i8 registers
    using VRegI8x4  = VecT<int8_t, 4>; // ...
    using VRegI8x8  = VecT<int8_t, 8>; //
    using VRegI8x16 = VecT<int8_t, 16>; //
    using VRegI8x32 = VecT<int8_t, 32>; // 32 i8 registers

    using VRegI32x1  = VecT<int32_t, 1>; // Single i32 register
    using VRegI32x2  = VecT<int32_t, 2>; // Two i32 registers
    using VRegI32x4  = VecT<int32_t, 4>; // ...
    using VRegI32x8  = VecT<int32_t, 8>; //
    using VRegI32x16 = VecT<int32_t, 16>; //
    using VRegI32x32 = VecT<int32_t, 32>; // 32 i32 registers

    using VRegF16x1  = VecT<float16_t, 1>; // Single f16 register
    using VRegF16x2  = VecT<float16_t, 2>; // Two f16 registers
    using VRegF16x4  = VecT<float16_t, 4>; // ...
    using VRegF16x8  = VecT<float16_t, 8>; //
    using VRegF16x16 = VecT<float16_t, 16>; //
    using VRegF16x32 = VecT<float16_t, 32>; // 32 f16 registers

    using VRegF32x1  = VecT<float32_t, 1>; // Single f32 register
    using VRegF32x2  = VecT<float32_t, 2>; // Two f32 registers
    using VRegF32x4  = VecT<float32_t, 4>; // ...
    using VRegF32x8  = VecT<float32_t, 8>; //
    using VRegF32x16 = VecT<float32_t, 16>; //
    using VRegF32x32 = VecT<float32_t, 32>; // 32 f32 registers

    using VRegF64x1  = VecT<float64_t, 1>; // Single f64 register
    using VRegF64x2  = VecT<float64_t, 2>; // Two f64 registers
    using VRegF64x4  = VecT<float64_t, 4>; // ...
    using VRegF64x8  = VecT<float64_t, 8>; //
    using VRegF64x16 = VecT<float64_t, 16>; //
    using VRegF64x32 = VecT<float64_t, 32>; // 32 f64 registers

    // Acc registers
    using AccRegI32x1  = VecT<int32_t, 1>;
    using AccRegI32x2  = VecT<int32_t, 2>;
    using AccRegI32x4  = VecT<int32_t, 4>;
    using AccRegI32x8  = VecT<int32_t, 8>;
    using AccRegI32x16 = VecT<int32_t, 16>;
    using AccRegI32x32 = VecT<int32_t, 32>;

    using AccRegF32x1  = VecT<float32_t, 1>;
    using AccRegF32x2  = VecT<float32_t, 2>;
    using AccRegF32x4  = VecT<float32_t, 4>;
    using AccRegF32x8  = VecT<float32_t, 8>;
    using AccRegF32x16 = VecT<float32_t, 16>;
    using AccRegF32x32 = VecT<float32_t, 32>;

    using AccRegF64x1  = VecT<float64_t, 1>;
    using AccRegF64x2  = VecT<float64_t, 2>;
    using AccRegF64x4  = VecT<float64_t, 4>;
    using AccRegF64x8  = VecT<float64_t, 8>;
    using AccRegF64x16 = VecT<float64_t, 16>;
    using AccRegF64x32 = VecT<float64_t, 32>;

    // Helper for string representations of types
    template <typename DataT>
    constexpr const char* dataTypeToString();

} // namespace rocwmma

#include "types_impl.hpp"

#include "types_ext.hpp"

#endif // ROCWMMA_TYPES_HPP
