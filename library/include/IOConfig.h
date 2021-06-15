#ifndef IO_CONFIG_H
#define IO_CONFIG_H

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
//#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOPack.h"
#include "IOTraits.h"
#include "IOUnpack.h"
#include "Layout.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Types.h"

// TODO: Remove when LLVM buffer commands are removed.
#define LLVM_BUFFER_BUG_WORKAROUND

// Optimal IO config
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig;

template <uint32_t BlockDim,
          uint32_t BlockK,
          typename DataT,
          typename DataLayout,
          template <uint32_t, uint32_t, typename, typename, uint32_t>
          class LoadLayout,
          uint32_t ElementsPerThread,
          uint32_t SpCount>
struct amdgcn_cooperative_load_DxK;

// Matrix A config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<matrix_a, BlockDim, BlockK, DataT, row_major>
{
    enum : uint32_t
    {
        ElementsPerThread = VecWidthTraits<BlockDim, BlockK, DataT>::MaxElementsPerThread
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config.
    // For fastest loading from global in row major, we should load by rows.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Col4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                row_major,
                                                GlobalLayout,
                                                ElementsPerThread>;

    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 row_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   row_major,
                                                   GlobalLayout,
                                                   ElementsPerThread>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    row_major,
                                                    GlobalLayout,
                                                    ElementsPerThread>;

    // Local data config.
    // After writing from global to LDS, load proper format for MFMA.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, row_major, LocalLayout, 1>;
};

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<matrix_a, BlockDim, BlockK, DataT, col_major>
{
    enum : uint32_t
    {
        ElementsPerThread = 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config.
    // For fastest loading from global in row major, we should load by rows.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Col4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                col_major,
                                                GlobalLayout,
                                                ElementsPerThread>;

    // template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    // using GlobalLayoutT = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 col_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   col_major,
                                                   GlobalLayout,
                                                   ElementsPerThread>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    col_major,
                                                    GlobalLayout,
                                                    ElementsPerThread>;

    // Local data config.
    // After writing from global to LDS, load proper format for MFMA.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLayout = Layout::Col4T<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, col_major, LocalLayout, 1>;
};

// Matrix B config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<matrix_b, BlockDim, BlockK, DataT, row_major>
{
    enum : uint32_t
    {
        ElementsPerThread = 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                row_major,
                                                GlobalLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 row_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   row_major,
                                                   GlobalLayout,
                                                   ElementsPerThread>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    row_major,
                                                    GlobalLayout,
                                                    ElementsPerThread>;

    // Local data config.
    // After writing from global to LDS, load proper format for MFMA.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, row_major, LocalLayout, 1>;
};

// Matrix B config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<matrix_b, BlockDim, BlockK, DataT, col_major>
{
    enum : uint32_t
    {
        ElementsPerThread = VecWidthTraits<BlockDim, BlockK, DataT>::MaxElementsPerThread
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                col_major,
                                                GlobalLayout,
                                                ElementsPerThread>;

    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 col_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   col_major,
                                                   GlobalLayout,
                                                   ElementsPerThread>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    col_major,
                                                    GlobalLayout,
                                                    ElementsPerThread>;

    // Local data config.
    // After writing from global to LDS, load proper format for MFMA.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, col_major, LocalLayout, 1>;
};

// Matrix C config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<accumulator, BlockDim, BlockK, DataT, row_major>
{
    enum : uint32_t
    {
        ElementsPerThread = 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                row_major,
                                                GlobalLayout,
                                                ElementsPerThread>;

    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 row_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;
};

template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<accumulator, BlockDim, BlockK, DataT, col_major>
{
    enum : uint32_t
    {
        ElementsPerThread = 4
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                col_major,
                                                GlobalLayout,
                                                ElementsPerThread>;

    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 col_major,
                                                 GlobalLayout,
                                                 ElementsPerThread>;
};

#endif // IO_CONFIG_H
