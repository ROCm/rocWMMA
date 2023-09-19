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

#ifndef ROCWMMA_DEVICE_TUPLE_TEST_HPP
#define ROCWMMA_DEVICE_TUPLE_TEST_HPP

#include <rocwmma/internal/io_traits.hpp>
#include <rocwmma/internal/layout.hpp>
#include <rocwmma/internal/tuple.hpp>
#include <rocwmma/rocwmma.hpp>

static constexpr uint32_t ERROR_VALUE   = 7u;
static constexpr uint32_t SUCCESS_VALUE = 0u;

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    __device__ static inline DataT get(VecT<DataT, VecSize> const& v, uint32_t idx)
    {
        return v.data[idx];
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool isTupleTest()
    {
        bool err = false;

        err |= std::is_tuple<int>::value;
        err |= !std::is_tuple<std::tuple<int, float>>::value;
        err |= std::is_tuple<const std::tuple<int, float>>::value;
        err |= std::is_tuple<std::tuple<int, float>&>::value;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool operatorMultTest()
    {
        bool err = false;

        auto srcTuple          = std::make_tuple(1, 2.0, 3u);
        auto expectHeadElement = std::make_tuple(2);
        auto resultHeadElement = std::operator_mult_impl(2, srcTuple, std::index_sequence<0>{});

        err |= expectHeadElement != resultHeadElement;

        auto expectTailElement = std::make_tuple(4.0, 6u);
        auto resultTailElement = std::operator_mult_impl(2, srcTuple, std::index_sequence<1, 2>{});

        err |= expectTailElement != resultTailElement;

        auto expectAllElement = std::make_tuple(2, 4.0, 6u);
        auto resultAllElement = 2 * srcTuple;

        err |= expectAllElement != resultAllElement;
        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool operatorAddTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expectLeftAddImplHeadElement = std::make_tuple(2);
        auto resultLeftAddImplHeadElement
            = std::operator_add_impl(1, srcTuple, std::index_sequence<0>{});
        err |= expectLeftAddImplHeadElement != resultLeftAddImplHeadElement;

        auto expectLeftAddImplTailElement = std::make_tuple(3.0, 4u);
        auto resultLeftAddImplTailElement
            = std::operator_add_impl(1, srcTuple, std::index_sequence<1, 2>{});
        err |= expectLeftAddImplTailElement != resultLeftAddImplTailElement;

        auto expectRightAddImplHeadElement = std::make_tuple(2);
        auto resultRightAddImplHeadElement
            = std::operator_add_impl(srcTuple, 1, std::index_sequence<0>{});
        err |= expectRightAddImplHeadElement != resultRightAddImplHeadElement;

        auto expectRightAddImplTailElement = std::make_tuple(3.0, 4u);
        auto resultRightAddImplTailElement
            = std::operator_add_impl(srcTuple, 1, std::index_sequence<1, 2>{});
        err |= expectRightAddImplTailElement != resultRightAddImplTailElement;

        auto expectLeftAddElement = std::make_tuple(2, 3.0, 4u);
        auto resultLeftAddElement = 1 + srcTuple;
        err |= expectLeftAddElement != resultLeftAddElement;

        auto expectRightAddElement = std::make_tuple(2, 3.0, 4u);
        auto resultRightAddElement = srcTuple + 1;
        err |= expectRightAddElement != resultRightAddElement;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool operatorSubTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        // auto expectLeftSubImplHeadElement = std::make_tuple(2);
        // auto resultLeftSubImplHeadElement = std::operator_sub_impl(1, srcTuple, std::index_sequence<0>{});
        // err |= expectLeftSubImplHeadElement != resultLeftSubImplHeadElement;

        // auto expectLeftSubImplTailElement = std::make_tuple(3.0, 4u);
        // auto resultLeftSubImplTailElement = std::operator_sub_impl(1, srcTuple, std::index_sequence<1, 2>{});
        // err |= expectLeftSubImplTailElement != resultLeftSubImplTailElement;

        auto expectRightSubImplHeadElement = std::make_tuple(0);
        auto resultRightSubImplHeadElement
            = std::operator_sub_impl(srcTuple, 1, std::index_sequence<0>{});
        err |= expectRightSubImplHeadElement != resultRightSubImplHeadElement;

        auto expectRightSubImplTailElement = std::make_tuple(1.0, 2u);
        auto resultRightSubImplTailElement
            = std::operator_sub_impl(srcTuple, 1, std::index_sequence<1, 2>{});
        err |= expectRightSubImplTailElement != resultRightSubImplTailElement;

        // auto expectLeftSubElement = std::make_tuple(2, 3.0, 4u);
        // auto resultLeftSubElement = 1 + srcTuple;
        // err |= expectLeftSubElement != resultLeftSubElement;

        auto expectRightSubElement = std::make_tuple(0, 1.0, 2u);
        auto resultRightSubElement = srcTuple - 1;
        err |= expectRightSubElement != resultRightSubElement;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool copyTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expect = std::make_tuple(1, 3u);
        auto result = rocwmma::copy_impl(srcTuple, std::index_sequence<0, 2>{});
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool popRightTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expect = std::make_tuple(1, 2.0);
        auto result = rocwmma::pop_right(srcTuple);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool popLeftTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expect = std::make_tuple(2.0, 3u);
        auto result = rocwmma::pop_left(srcTuple);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool getFirstTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expect = 1;
        auto result = rocwmma::get_first(srcTuple);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool getLastTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2.0, 3u);

        auto expect = 3u;
        auto result = rocwmma::get_last(srcTuple);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool reverseTest()
    {
        bool err = false;

        auto srcTuple = std::make_tuple(1, 2, 3);

        auto expect = std::make_tuple(3, 2, 1);
        auto result = rocwmma::reverse(srcTuple);
        err |= expect != result;

        return err;
    }

    /**
     * More Details about flatten and inflate
     * https://coderwall.com/p/fzni3g/bidirectional-translation-between-1d-and-3d-arrays
     */
    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool flattenCoordRightTest()
    {
        bool err = false;

        auto srcCoord = std::make_tuple(2, 3, 5, 7);
        auto srcDims  = std::make_tuple(3, 5, 7, 11);

        /**
         * | c      | 2 | 3  | 5  | 7   |
         * | d      | 3 | 5  | 7  | 11  |
         * | mul    | 1 | 3  | 15 | 105 |
         * | result | 2 | 11 | 86 | 821 |
         */
        auto expect = 821;
        auto result = rocwmma::flatten_coord_right(srcCoord, srcDims);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool flattenCoordLeftTest()
    {
        bool err = false;

        auto srcCoord = std::make_tuple(2, 3, 5, 7);
        auto srcDims  = std::make_tuple(3, 5, 7, 11);

        /**
         * | c      | 7  | 5  | 3   | 2    |
         * | d      | 11 | 7  | 5   | 3    |
         * | mul    | 1  | 11 | 77  | 385  |
         * | result | 7  | 62 | 293 | 1063 |
         */
        auto expect = 1063;
        auto result = rocwmma::flatten_coord_left(srcCoord, srcDims);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool inflateCoordRightTest()
    {
        bool err = false;

        auto srcFlatCoord = 821;
        auto srcDims      = std::make_tuple(3, 5, 7, 11);

        /**
         * | c      | 821 | 821 | 821 | 821 |
         * | d      | 3   | 5   | 7   | 11  |
         * | div    | 1   | 3   | 15  | 105 |
         * | result | 2   | 3   | 5   | 7   |
         */
        auto expect = std::make_tuple(2, 3, 5, 7);
        auto result = rocwmma::inflate_coord_right(srcFlatCoord, srcDims);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool inflateCoordLeftTest()
    {
        bool err = false;

        auto srcFlatCoord = 1063;
        auto srcDims      = std::make_tuple(3, 5, 7, 11);

        /**
         * | c               | 1063 | 1063 | 1063 | 1063 |
         * | d               | 11   | 7    | 5    | 3    |
         * | div             | 1    | 11   | 77   | 385  |
         * | result          | 7    | 5    | 3    | 2    |
         * | reversed result | 2    | 3    | 5    | 7    |
         */
        auto expect = std::make_tuple(2, 3, 5, 7);
        auto result = rocwmma::inflate_coord_left(srcFlatCoord, srcDims);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __device__ static inline bool toMatrixSpaceTest()
    {
        bool err = false;

        auto srcStrides      = std::make_tuple(2, 3, 5, 7);
        auto srcStrideCounts = std::make_tuple(3, 5, 7, 11);

        /**
         * | stride | 2 | 3  | 5  | 7   |
         * | count  | 3 | 5  | 7  | 11  |
         * | result | 6 | 21 | 56 | 133 |
         */
        auto expect = 133;
        auto result = rocwmma::to_matrix_space(srcStrides, srcStrideCounts);
        err |= expect != result;

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __global__ void tupleTest(uint32_t     m,
                              uint32_t     n,
                              DataT const* in,
                              DataT*       out,
                              uint32_t     ld,
                              DataT        param1,
                              DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        err = err ? err : isTupleTest<DataT, VecSize>();
        err = err ? err : operatorMultTest<DataT, VecSize>();
        err = err ? err : operatorAddTest<DataT, VecSize>();
        err = err ? err : operatorSubTest<DataT, VecSize>();
        err = err ? err : copyTest<DataT, VecSize>();
        err = err ? err : popLeftTest<DataT, VecSize>();
        err = err ? err : popRightTest<DataT, VecSize>();
        err = err ? err : getFirstTest<DataT, VecSize>();
        err = err ? err : getLastTest<DataT, VecSize>();
        err = err ? err : reverseTest<DataT, VecSize>();
        err = err ? err : flattenCoordRightTest<DataT, VecSize>();
        err = err ? err : flattenCoordLeftTest<DataT, VecSize>();
        err = err ? err : inflateCoordRightTest<DataT, VecSize>();
        err = err ? err : inflateCoordLeftTest<DataT, VecSize>();
        err = err ? err : toMatrixSpaceTest<DataT, VecSize>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_TUPLE_TEST_HPP
