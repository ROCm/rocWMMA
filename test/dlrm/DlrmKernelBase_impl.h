/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#ifndef DLRM_KERNEL_BASE_IMPL_H
#define DLRM_KERNEL_BASE_IMPL_H
#define __HIP_PLATFORM_HCC true

#include "DlrmKernelBase.h"
#include <cstdlib>
#include <iostream>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

#include "Common.hpp"
#include "Performance.h"

// Library includes
#include "Constants.h"
#include "Utils.h"

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::DlrmKernelBase()
{
    reset();
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::~DlrmKernelBase();
{
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::gridDim() const
{
    return dim3(ceilDiv(mM, BlockM * mTBlockX / AMDGCN_WAVE_SIZE), ceilDiv(mN, BlockN * mTBlockY));
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::blockDim() const
{
    return dim3(mTBlockX, mTBlockY);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::checkDevice() const
{
    auto deviceArch = DeviceInfo::instance()->getGcnArch();
    return (deviceArch != DeviceInfo::UNKNOWN
            && !(deviceArch == DeviceInfo::GFX908 && std::is_same<DataT, float64_t>::value));
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::checkSizes() const
{
    return (mM >= (BlockM * mTBlockX / AMDGCN_WAVE_SIZE) && mN >= (BlockN * mTBlockY)
            && mK >= BlockK);
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::reset()
{
    mTBlockX = mTBlockY = 0;
    mM = mN = mK = 0;

    mRunFlag = true;

    mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
    mElapsedTimeMs = mEfficiency = 0.0;
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
DlrmKernelBase<BlockM, BlockN, BlockK, DataT>::setup(ProblemParams const& problem)
{
    // Reset the flags in case of multiple runs
    mRunFlag = true;

    // Format incoming problem parameters
    std::tie(mTBlockX, mTBlockY)
        = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.threadBlockSize)),
                   static_cast<uint32_t const&>(std::get<1>(problem.threadBlockSize)));
    std::tie(mM, mN, mK) = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                                    static_cast<uint32_t const&>(std::get<2>(problem.problemSize)));

    mRunFlag &= checkDevice();
    mRunFlag &= checkSizes();

    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();

        int         fp    = (std::is_same<DataT, float32_t>::value) ? 32 : 16;
        std::string fpStr = std::to_string(fp);

        // Determine whether to run forward or backward pass
        isBwd = problem.isBwd;

        // Initialize matrix storage
        if(!isBwd)
            dataInstance->resizeFwdStorage(problem.fwdDataSize);
        else
            dataInstance->resizeBwdStorage(problem.bwdDataSize);

        // Initialize matrix data on host and transfer to device
        // (check for null pointer for reading reference data only once)
        if(!isBwd)
        {
            int fdInput     = open("dlrmData/input_fp" + fpStr, O_RDONLY);
            int fdOutputRef = open("dlrmData/output_fp" + fpStr, O_RDONLY);

            pread(fdInput,
                  dataInstance->hostInput().get(),
                  std::get<0>(problem.fwdDataSize) * sizeof(DataT),
                  0);
            pread(fdOutputRef,
                  dataInstance->hostOutputRef().get(),
                  std::get<2>(problem.fwdDataSize) * sizeof(DataT),
                  0);

            dataInstance->copyHostToDeviceFwdAll();
        }
        else
        {
            int fdInput            = open("dlrmData/input_fp" + fpStr, O_RDONLY);
            int fdUpstreamGrad     = open("dlrmData/input_grad_fp" + fpStr, O_RDONLY);
            int fdGradRef          = open("dlrmData/output_input_grad_fp" + fpStr, O_RDONLY);
            int fdBottomMlpGradRef = open("dlrmData/output_mlp_input_grad_fp" + fpStr, O_RDONLY);

            pread(fdInput,
                  dataInstance->hostInput().get(),
                  std::get<0>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdUpstreamGrad,
                  dataInstance->hostUpstreamGrad().get(),
                  std::get<1>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdGradRef,
                  dataInstance->hostGradRef().get(),
                  std::get<3>(problem.bwdDataSize) * sizeof(DataT),
                  0);
            pread(fdBottomMlpGradRef,
                  dataInstance->hostBottomMlpGradRef().get(),
                  std::get<5>(problem.bwdDataSize) * sizeof(DataT),
                  0);

            dataInstance->copyHostToDeviceBwdAll();
        }
    }
}

template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize typename DataT>
// only pass variable parameters as template params
// fwd
warps_per_threadblock, // hard coded
    threadblock_size, // hard coded
    M_BLOCKS, // hard coded
    K_BLOCKS, // hard coded
    SMEM_STRIDE, // hard coded
    SMEM_STRIDE_ACC, // hard coded
    kWarpSize, // shared hard coded
    kWarpSizeLog2, // shared hard coded
    kTileDim, // shared hard coded for fwd, 16/32 for bwd
    kTileDimLog2 // shared hard coded for fwd, ^

        // bwd
        kWarpsPerBlock, // hard coded
    kNumThreads, // hard coded
    kRowTilesPerStep, // hard coded for fwd, 2/1 for bwd
    kColTilesPerStep, // hard coded
    kWarpSize, // shared hard coded
    kWarpSizeLog2, // shared hard coded
    kTileDim, // shared hard coded for fwd, 16/32 for bwd
    kTileDimLog2 // shared hard coded for fwd, ^
    // setup nonaligned kernel calls

    DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>::exec()
{
    if(mRunFlag)
    {
        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

        auto& dataInstance = DataStorage::instance();

        if(!isBwd)
        {
            // num tiles
            uint num_row_tiles = (NUM_ROWS + kTileDim - 1) >> kTileDimLog2;
            uint num_col_tiles = (NUM_COLS + kTileDim - 1) >> kTileDimLog2;

            // number of rows and columns after padding
            uint num_rows_after_padding = kTileDim << 1;
            uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

            uint num_row_steps = num_row_tiles / kRowTilesPerStep;
            uint num_col_steps = num_col_tiles / kColTilesPerStep;

            // *remove const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
            // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
            const uint smem_rows_per_warp      = M_BLOCKS << 4;
            const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
            // *remove const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
            const uint smem_elems_per_warp_acc
                = M_BLOCKS * kTileDim * SMEM_STRIDE_ACC * 2; // output in FP32
            const uint smem_elems_per_warp = (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
                                                 ? smem_elems_per_warp_mat
                                                 : smem_elems_per_warp_acc;
            uint       output_size         = NUM_COLS + (NUM_ROWS * (NUM_ROWS - 1) >> 1) + PAD;

            bool float4_predicate = !((NUM_COLS & 7) || (output_size & 7));

            if(float4_predicate)
            {
                hipExtLaunchKernelGGL((kernelFwdImpl()),
                                      (gridDim()),
                                      (blockDim()),
                                      (ldsUsage()),
                                      0,
                                      startEvent,
                                      stopEvent,
                                      0,
                                      dataInstance->deviceInput().get(),
                                      dataInstance->deviceOutput().get(),
                                      BATCH_SIZE,
                                      NUM_ROWS,
                                      NUM_COLS,
                                      num_rows_after_padding,
                                      num_cols_after_padding,
                                      smem_elems_per_warp,
                                      smem_rows_per_warp,
                                      output_size,
                                      num_row_stemps,
                                      num_col_steps,
                                      PAD);
            }
            else
            {
                hipExtLaunchKernelGGL((kernelFwdNonAlignedImpl()),
                                      (gridDim()),
                                      (blockDim()),
                                      (ldsUsage()),
                                      0,
                                      startEvent,
                                      stopEvent,
                                      0,
                                      dataInstance->deviceInput().get(),
                                      dataInstance->deviceOutput().get(),
                                      BATCH_SIZE,
                                      NUM_ROWS,
                                      NUM_COLS,
                                      num_rows_after_padding,
                                      num_cols_after_padding,
                                      smem_elems_per_warp,
                                      smem_rows_per_warp,
                                      output_size,
                                      num_row_stemps,
                                      num_col_steps,
                                      PAD);
            }
        }
        else
        {
            const uint mem_skew_size      = 8;
            const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;

            uint input_dbytes = amp_train ? sizeof(half) : sizeof(float);

            uint row_tiles_per_step = NUM_ROWS > kTileDim ? kRowTilesPerStep : 1;

            // num tiles
            uint num_row_tiles = (NUM_ROWS + kTileDim - 1) >> kTileDimLog2;
            uint num_col_tiles = (NUM_COLS + kTileDim - 1) >> kTileDimLog2;

            // number of rows and columns after padding
            uint num_rows_after_padding = num_row_tiles << kTileDimLog2;
            uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

            // 2D ugrad size and stride
            uint interaction_ugrad_2D_stride = num_rows_after_padding + mem_skew_size;
            uint interaction_ugrad_2D_size_elems
                = num_rows_after_padding * interaction_ugrad_2D_stride;
            uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * input_dbytes;

            // 1D ugrad size
            uint interaction_ugrad_size              = NUM_ROWS * (NUM_ROWS - 1) >> 1;
            uint interaction_ugrad_size_with_padding = interaction_ugrad_size + PAD;

            // in_out place size and stride
            uint input_stride     = num_cols_after_padding + mem_skew_size;
            uint input_size_elems = num_rows_after_padding * input_stride;
            uint input_size_bytes = input_size_elems * input_dbytes;

            // sample size
            uint sample_size = NUM_ROWS * NUM_COLS;

            // output size
            uint output_size_elems = kTileDim * kTileDim * kRowTilesPerStep * kColTilesPerStep;
            uint output_size_bytes = output_size_elems * sizeof(float);

            // staging area size
            uint staging_area_size_bytes = output_size_bytes > interaction_ugrad_2D_size_bytes
                                               ? output_size_bytes
                                               : interaction_ugrad_2D_size_bytes;

            // Shared memory size
            uint wmma_smem_byte = num_rows_after_padding * num_rows_after_padding * input_dbytes;
            uint shared_mem_per_warp_size_byte
                = input_size_bytes + staging_area_size_bytes + wmma_smem_byte;
            uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

            uint num_blocks    = (BATCH_SIZE + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
            uint num_row_steps = num_row_tiles / row_tiles_per_step;
            uint num_col_steps = num_col_tiles / kColTilesPerStep;

            bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (NUM_COLS & 7));

            if(float4_predicate)
            {
                hipExtLaunchKernelGGL((kernelBwdImpl()),
                                      (gridDim()),
                                      (blockDim()),
                                      (ldsUsage()),
                                      0,
                                      startEvent,
                                      stopEvent,
                                      0,
                                      dataInstance->deviceInput().get(),
                                      dataInstance->deviceUpstreamGrad().get(),
                                      dataInstance->deviceGrad().get(),
                                      dataInstance->deviceBottomMlpGrad().get(),
                                      BATCH_SIZE,
                                      NUM_ROWS,
                                      NUM_COLS,
                                      num_rows_after_padding,
                                      num_cols_after_padding,
                                      sample_size,
                                      interaction_ugrad_size,
                                      interaction_ugrad_size_with_padding,
                                      interaction_ugrad_2D_size_elems,
                                      interaction_ugrad_2D_stride,
                                      input_size_elems,
                                      input_stride,
                                      num_row_steps,
                                      num_col_steps,
                                      row_tiles_per_step,
                                      shared_mem_per_warp_size_byte);
            }
            else
            {
                hipExtLaunchKernelGGL((kernelBwdNonAlignedImpl()),
                                      (gridDim()),
                                      (blockDim()),
                                      (ldsUsage()),
                                      0,
                                      startEvent,
                                      stopEvent,
                                      0,
                                      dataInstance->deviceInput().get(),
                                      dataInstance->deviceUpstreamGrad().get(),
                                      dataInstance->deviceGrad().get(),
                                      dataInstance->deviceBottomMlpGrad().get(),
                                      BATCH_SIZE,
                                      NUM_ROWS,
                                      NUM_COLS,
                                      num_rows_after_padding,
                                      num_cols_after_padding,
                                      sample_size,
                                      interaction_ugrad_size,
                                      interaction_ugrad_size_with_padding,
                                      interaction_ugrad_2D_size_elems,
                                      interaction_ugrad_2D_stride,
                                      input_size_elems,
                                      input_stride,
                                      num_row_steps,
                                      num_col_steps,
                                      row_tiles_per_step,
                                      shared_mem_per_warp_size_byte);
            }
        }

        // Calculate efficiency
        auto& deviceInfo             = DeviceInfo::instance();
        auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<InputT>();

        mElapsedTimeMs        = float64_t(timeMs);
        mTotalGFlops          = calculateGFlops(mM, mN, mK);
        mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs);
        mEfficiency           = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;
    }
}

DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>::validateResults()
{
#ifdef WMMA_VALIDATE_TESTS
    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();
    }
}

DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>::reportResults() {}

DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>::tearDown() {}

DlrmKernelBase<BlockM, BlockN, BlockK, TileSize, DataT>::debugStr() {}

#endif // DLRM_KERNEL_BASE_IMPL_H
