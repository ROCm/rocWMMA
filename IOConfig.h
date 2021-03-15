#ifndef IO_CONFIG_H
#define IO_CONFIG_H

#include "BufferLoad.h"
#include "BufferStore.h"
#include "Constants.h"
#include "CoopStore.h"
#include "IOPack.h"
#include "IOTraits.h"
#include "IOUnpack.h"
#include "Layout.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Types.h"

#define LLVM_BUFFER_BUG_WORKAROUND

template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct IOConfig;

// Matrix A config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct IOConfig<matrix_a, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        ElementsPerThread = std::is_same<DataLayout, row_major>::value ? 4 : 1
    };

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLoadLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalStoreLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

#ifdef LLVM_BUFFER_BUG_WORKAROUND
    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#else
    using GlobalLoader = amdgcn_buffer_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_buffer_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#endif

    // Local LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLoadLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalStoreLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                               BlockK,
                                               DataT,
                                               DataLayout,
                                               LocalLoadLayout,
                                               ElementsPerThread>;
    using LocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                LocalStoreLayout,
                                                ElementsPerThread>;

    // Coop LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalLoadLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalStoreLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using CoopLocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   DataLayout,
                                                   CoopLocalLoadLayout,
                                                   ElementsPerThread>;
    using CoopLocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    DataLayout,
                                                    CoopLocalStoreLayout,
                                                    ElementsPerThread>;

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;
};

// Matrix B config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct IOConfig<matrix_b, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        ElementsPerThread = std::is_same<DataLayout, col_major>::value ? 4 : 1
    };

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLoadLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalStoreLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

#ifdef LLVM_BUFFER_BUG_WORKAROUND
    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#else
    using GlobalLoader = amdgcn_buffer_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_buffer_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#endif

    // Local LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLoadLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalStoreLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                               BlockK,
                                               DataT,
                                               DataLayout,
                                               LocalLoadLayout,
                                               ElementsPerThread>;
    using LocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                LocalStoreLayout,
                                                ElementsPerThread>;

    // Coop LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalLoadLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalStoreLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using CoopLocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   DataLayout,
                                                   CoopLocalLoadLayout,
                                                   ElementsPerThread>;
    using CoopLocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    DataLayout,
                                                    CoopLocalStoreLayout,
                                                    ElementsPerThread>;

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;
};

// Matrix C/D config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct IOConfig<accumulator, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        ElementsPerThread = 1
    };

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLoadLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalStoreLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

#ifdef LLVM_BUFFER_BUG_WORKAROUND
    using GlobalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#else
    using GlobalLoader = amdgcn_buffer_load_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                GlobalLoadLayout,
                                                ElementsPerThread>;
    using GlobalStorer = amdgcn_buffer_store_DxK<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 DataLayout,
                                                 GlobalStoreLayout,
                                                 ElementsPerThread>;
#endif

    // Local LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLoadLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalStoreLayout = Layout::Row4T<BlkDim, BlkK, DT, DL, EPT>;

    using LocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                               BlockK,
                                               DataT,
                                               DataLayout,
                                               LocalLoadLayout,
                                               ElementsPerThread>;
    using LocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                BlockK,
                                                DataT,
                                                DataLayout,
                                                LocalStoreLayout,
                                                ElementsPerThread>;

    // Coop LDS data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalLoadLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using CoopLocalStoreLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

    using CoopLocalLoader = amdgcn_opaque_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   DataLayout,
                                                   CoopLocalLoadLayout,
                                                   ElementsPerThread>;
    using CoopLocalStorer = amdgcn_opaque_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    DataLayout,
                                                    CoopLocalStoreLayout,
                                                    ElementsPerThread>;

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;
};

// Optimal config for no sharing between workgroups
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig;

// Matrix A config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT>
struct OptConfig<matrix_a, BlockDim, BlockK, DataT, row_major>
{
    enum : uint32_t
    {
        ElementsPerThread = 4 * PackTraits<DataT>::PackRatio
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config.
    // For fastest loading from global in row major, we should load by rows.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

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

    using CoopStorer = amdgcn_cooperative_store_dword_DxK<matrix_a,
                                                          BlockDim,
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
    using GlobalLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

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

    using CoopStorer = amdgcn_cooperative_store_dword_DxK<matrix_a,
                                                          BlockDim,
                                                          BlockK,
                                                          DataT,
                                                          col_major,
                                                          GlobalLayout,
                                                          ElementsPerThread>;

    // Local data config.
    // After writing from global to LDS, load proper format for MFMA.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using LocalLayout = Layout::Col<BlkDim, BlkK, DT, DL, EPT>;

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
    using GlobalLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

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

    using CoopStorer = amdgcn_cooperative_store_dword_DxK<matrix_b,
                                                          BlockDim,
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
        ElementsPerThread = 4 * PackTraits<DataT>::PackRatio
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

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

    using CoopStorer = amdgcn_cooperative_store_dword_DxK<matrix_b,
                                                          BlockDim,
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
        ElementsPerThread = 4 * PackTraits<DataT>::PackRatio
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, ElementsPerThread>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::Row<BlkDim, BlkK, DT, DL, EPT>;

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
