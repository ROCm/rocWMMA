#ifndef IO_CONFIG_H
#define IO_CONFIG_H

#include "Layout.h"
#include "Types.h"

// Selector for buffer I/O layout based on MatrixT and DataLayout.
// Possible MatrixT types:
// matrix_a
// matrix_b
// accumulator
// Possible DataLayout Types:
// row_major
// col_major
template <typename MatrixT, typename DataLayout>
struct BufferConfig;

// Matrix A loads matrix columns of size BlockDim in the K direction
template <>
struct BufferConfig<matrix_a, row_major>
{
    // Row major config for A column loading has no contiguous neighbours
    enum : uint32_t 
    {
        ElementsPerThread = 1// MAX_ELEMENTS_PER_THREAD
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Col<BlockDim, BlockK, DataT, row_major, 1>;
};

template <>
struct BufferConfig<matrix_a, col_major>
{
    // Col major config for A column loading has many contiguous neighbours
    enum : uint32_t 
    {
        ElementsPerThread = 1//MAX_ELEMENTS_PER_THREAD
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Col<BlockDim, BlockK, DataT, col_major, ElementsPerThread>;
};

template <>
struct BufferConfig<matrix_b, row_major>
{
    // Row major config for B row loading has many contiguous neighbours
    enum : uint32_t 
    {
        ElementsPerThread = 1// MAX_ELEMENTS_PER_THREAD
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Row<BlockDim, BlockK, DataT, row_major, ElementsPerThread>;
};

template <>
struct BufferConfig<matrix_b, col_major>
{
    // Col major config for B row loading has no contiguous neighbours
    enum : uint32_t 
    {
        ElementsPerThread = 1// MAX_ELEMENTS_PER_THREAD
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Row<BlockDim, BlockK, DataT, col_major, 1>;
};

// Accumulator loads matrix rows of size BlockDim in the K direction,
// with the rows transposed each group of 4 registers.
template <typename DataLayout>
struct BufferConfig<accumulator, DataLayout>
{
    // For now, use ElementCount of 1 until indexing is adjusted
    enum : uint32_t 
    {
        ElementsPerThread = 1
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Row4T<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;
};


// Selector for local I/O layout based on MatrixT
// Possible MatrixT types:
// matrix_a
// matrix_b
// accumulator
template <typename MatrixT, typename DataLayout>
struct LocalConfig
{
    // TODO: For now, just stick with 1 element per thread until x2, x3, x4 are implemented
    enum : uint32_t 
    {
        ElementsPerThread = 1
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
    using LayoutT = Layout::Row<BlockDim, BlockK, DataT, DataLayout, ElementsPerThread>;
};

#endif // IO_CONFIG_H
