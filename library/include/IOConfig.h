#ifndef IO_CONFIG_H
#define IO_CONFIG_H

#include "Constants.h"
#include "CoopLoad.h"
#include "CoopStore.h"
#include "IOPack.h"
#include "IOTraits.h"
#include "IOUnpack.h"
#include "Layout.h"
#include "OpaqueLoad.h"
#include "OpaqueStore.h"
#include "Types.h"

// Optimal IO config
template <typename MatrixT, uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig;

// Matrix A config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig<matrix_a, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        MaxVectorWidth = VecWidthTraits<BlockDim, BlockK, DataT>::MaxElementsPerThread,
        VectorWidth    = std::is_same<DataLayout, row_major>::value ? MaxVectorWidth : 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config.
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::ColNT<BlkDim, BlkK, DT, DL, EPT, MaxVectorWidth>;

    using GlobalLoader
        = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;

    using GlobalStorer
        = amdgcn_opaque_store_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   DataLayout,
                                                   GlobalLayout,
                                                   VectorWidth>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    DataLayout,
                                                    GlobalLayout,
                                                    VectorWidth>;
};

// Matrix B config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig<matrix_b, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        MaxVectorWidth = VecWidthTraits<BlockDim, BlockK, DataT>::MaxElementsPerThread,
        VectorWidth    = std::is_same<DataLayout, col_major>::value ? MaxVectorWidth : 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::RowNT<BlkDim, BlkK, DT, DL, EPT, MaxVectorWidth>;

    using GlobalLoader
        = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;
    using GlobalStorer
        = amdgcn_opaque_store_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;

    using CoopLoader = amdgcn_cooperative_load_DxK<BlockDim,
                                                   BlockK,
                                                   DataT,
                                                   DataLayout,
                                                   GlobalLayout,
                                                   VectorWidth>;

    using CoopStorer = amdgcn_cooperative_store_DxK<BlockDim,
                                                    BlockK,
                                                    DataT,
                                                    DataLayout,
                                                    GlobalLayout,
                                                    VectorWidth>;
};

// Matrix C / D config
template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename DataLayout>
struct OptConfig<accumulator, BlockDim, BlockK, DataT, DataLayout>
{
    enum : uint32_t
    {
        MaxVectorWidth = 4, // Actual output of the mfma hardware
        VectorWidth    = std::is_same<DataLayout, col_major>::value ? MaxVectorWidth : 1
    };

    // Other IO configs
    using IOTraits = amdgcn_io_traits<BlockDim, BlockK, DataT, VectorWidth>;
    using Packer   = Pack<DataT, IOTraits::UnpackedRegisterCount>;
    using Unpacker = Unpack<DataT, IOTraits::PackedRegisterCount>;

    // Global data config
    template <uint32_t BlkDim, uint32_t BlkK, typename DT, typename DL, uint32_t EPT>
    using GlobalLayout = Layout::RowNT<BlkDim, BlkK, DT, DL, EPT, MaxVectorWidth>;

    using GlobalLoader
        = amdgcn_opaque_load_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;

    using GlobalStorer
        = amdgcn_opaque_store_DxK<BlockDim, BlockK, DataT, DataLayout, GlobalLayout, VectorWidth>;
};

#endif // IO_CONFIG_H
