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

#include "DlrmKernelBase.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <gtest/gtest.h>

#include "Common.hpp"
#include "Performance.h"
#include "common.h"

// Library includes
#include "Constants.h"
#include "Utils.h"

template <uint32_t TileSize, typename DataT>
DlrmKernelBase<TileSize, DataT>::DlrmKernelBase()
{
    reset();
}

template <uint32_t TileSize, typename DataT>
DlrmKernelBase<TileSize, DataT>::~DlrmKernelBase()
{
}

template <uint32_t TileSize, typename DataT>
uint32_t DlrmKernelBase<TileSize, DataT>::ldsUsage() const
{
    // return 0;
    if(!isBwd)
    {
        const uint smem_rows_per_warp      = M_BLOCKS << 4;
        const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
        const uint smem_elems_per_warp_acc = M_BLOCKS * kTileDim * SMEM_STRIDE_ACC * 2;
        const uint smem_elems_per_warp     = (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
                                                 ? smem_elems_per_warp_mat
                                                 : smem_elems_per_warp_acc;

        return warps_per_threadblock * smem_elems_per_warp * sizeof(DataT);
    }
    else
    {
        const uint TileSizeLog2 = Log2<TileSize>::value;

        uint input_dbytes  = sizeof(DataT);
        uint num_row_tiles = (NUM_ROWS + TileSize - 1) >> TileSizeLog2;
        uint num_col_tiles = (NUM_COLS + TileSize - 1) >> TileSizeLog2;

        uint num_rows_after_padding = num_row_tiles << TileSizeLog2;
        uint num_cols_after_padding = num_col_tiles << TileSizeLog2;

        uint interaction_ugrad_2D_stride     = num_rows_after_padding + MEM_SKEW_SIZE;
        uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;
        uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * input_dbytes;

        uint input_stride     = num_cols_after_padding + MEM_SKEW_SIZE;
        uint input_size_elems = num_rows_after_padding * input_stride;
        uint input_size_bytes = input_size_elems * input_dbytes;

        uint output_size_elems
            = TileSize * TileSize * kColTilesPerStep * (TileSize == 32 ? 1 : kRowTilesPerStep);
        uint output_size_bytes = output_size_elems * sizeof(float);

        uint staging_area_size_bytes = output_size_bytes > interaction_ugrad_2D_size_bytes
                                           ? output_size_bytes
                                           : interaction_ugrad_2D_size_bytes;

        uint wmma_smem_byte = num_rows_after_padding * num_rows_after_padding * input_dbytes;
        uint shared_mem_per_warp_size_byte
            = input_size_bytes + staging_area_size_bytes + wmma_smem_byte;
        uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

        return kWarpsPerBlock * shared_mem_per_warp_size_byte;
    }
}

template <uint32_t TileSize, typename DataT>
dim3 DlrmKernelBase<TileSize, DataT>::gridDim() const
{
    if(!isBwd)
    {
        return dim3((BATCH_SIZE + warps_per_threadblock - 1) / warps_per_threadblock);
    }
    else
        return dim3((BATCH_SIZE + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2);
}

template <uint32_t TileSize, typename DataT>
dim3 DlrmKernelBase<TileSize, DataT>::blockDim() const
{
    // return dim3(mTBlockX, mTBlockY);
    if(!isBwd)
    {
        return dim3(threadblock_size);
    }
    else
        return dim3(kNumThreads);
}

template <uint32_t TileSize, typename DataT>
bool DlrmKernelBase<TileSize, DataT>::checkDevice() const
{
    auto deviceArch = DeviceInfo::instance()->getGcnArch();
    return (deviceArch != DeviceInfo::UNKNOWN
            && !(deviceArch == DeviceInfo::GFX908 && std::is_same<DataT, float64_t>::value));
}

template <uint32_t TileSize, typename DataT>
bool DlrmKernelBase<TileSize, DataT>::checkSizes() const
{
    // return (mM >= (BlockM * mTBlockX / AMDGCN_WAVE_SIZE) && mN >= (BlockN * mTBlockY)
    //         && mK >= BlockK);
    return true;
}

template <uint32_t TileSize, typename DataT>
bool DlrmKernelBase<TileSize, DataT>::checkLds() const
{
    return ldsUsage() <= LDS_MAX_BYTES;
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::reset()
{
    mM = mN = mK = 0;

    mRepeats =
#ifdef WMMA_VALIDATION_TESTS
        1;
#else
        5;
#endif

    mRunFlag = true;

    mTotalGFlops = mMeasuredGFlopsPerSec = 0.0;
    mElapsedTimeMs = mEfficiency = 0.0;
}

template <uint32_t TileSize, typename DataT>
std::ostream& DlrmKernelBase<TileSize, DataT>::printHeader(std::ostream& stream) const
{
    return stream << "TileSize, "
                  << "DataT, "
                  << "Bwd, "
#if defined(WMMA_VALIDATION_TESTS)
                  << "numElements, "
                  << "maxAbsoluteDiff, "
                  << "maxRelativeDiff, "
                  << "tolerance, "
#endif
                  << "elapsedMs, "
                  << "GFlops, "
                  << "GFlops/s, "
                  << "Efficiency(%)" << std::endl;
}

template <uint32_t TileSize, typename DataT>
std::ostream& DlrmKernelBase<TileSize, DataT>::printKernel(std::ostream& stream) const
{
    return stream << TileSize << ", " << dataTypeToString<DataT>() << ", " << isBwd << ", "

#if defined(WMMA_VALIDATION_TESTS)
                  << mValidationResult.numElements << ", " << mValidationResult.maxAbsoluteDiff
                  << ", " << mValidationResult.maxRelativeDiff << ", "
                  << mValidationResult.tolerance << ", "
#endif
                  << mElapsedTimeMs << ", " << mTotalGFlops << ", " << mMeasuredGFlopsPerSec << ", "
                  << mEfficiency << ", "
#if defined(WMMA_VALIDATION_TESTS)
                  << (mValidationResult.pass ? "PASSED" : "FAILED")
#else
                  << "BENCH"
#endif // WMMA_VALIDATION_TESTS
                  << std::endl;
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::setup(ProblemParams const& problem)
{
    // Reset the flags in case of multiple runs
    mRunFlag = true;

    // Format incoming problem parameters
    std::tie(mM, mN, mK)
        = std::tie(static_cast<uint32_t const&>(std::get<0>(problem.problemSize)),
                   static_cast<uint32_t const&>(std::get<1>(problem.problemSize)),
                   static_cast<uint32_t const&>(std::get<2>(problem.problemSize) * BATCH_SIZE));

    mRunFlag &= checkDevice();
    mRunFlag &= checkSizes();
    mRunFlag &= checkLds();

    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();

        int         fp    = (std::is_same<DataT, float32_t>::value) ? 32 : 16;
        std::string fpStr = std::to_string(fp);
        size_t      result;

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
            std::string inputFile     = "dlrmData/input_fp" + fpStr;
            std::string outputRefFile = "dlrmData/output_fp" + fpStr;

            FILE* inputFp = fopen(inputFile.c_str(), "rb");
            checkFileOpen(inputFp, inputFile);
            FILE* outputRefFp = fopen(outputRefFile.c_str(), "rb");
            checkFileOpen(outputRefFp, outputRefFile);

            result = fread(dataInstance->hostInput().get(),
                           sizeof(DataT),
                           std::get<0>(problem.fwdDataSize),
                           inputFp);
            checkFileSize(inputFile, result, std::get<0>(problem.fwdDataSize));

            result = fread(dataInstance->hostOutputRef().get(),
                           sizeof(DataT),
                           std::get<2>(problem.fwdDataSize),
                           outputRefFp);
            checkFileSize(outputRefFile, result, std::get<2>(problem.fwdDataSize));

            dataInstance->copyHostToDeviceFwdAll();
        }
        else
        {
            std::string inputFile            = "dlrmData/input_fp" + fpStr;
            std::string upstreamGradFile     = "dlrmData/input_grad_fp" + fpStr;
            std::string gradRefFile          = "dlrmData/output_input_grad_fp" + fpStr;
            std::string bottomMlpGradRefFile = "dlrmData/output_mlp_input_grad_fp" + fpStr;

            FILE* inputFp = fopen(inputFile.c_str(), "rb");
            checkFileOpen(inputFp, inputFile);
            FILE* upstreamGradFp = fopen(upstreamGradFile.c_str(), "rb");
            checkFileOpen(upstreamGradFp, upstreamGradFile);
            FILE* gradRefFp = fopen(gradRefFile.c_str(), "rb");
            checkFileOpen(gradRefFp, gradRefFile);
            FILE* bottomMlpGradRefFp = fopen(bottomMlpGradRefFile.c_str(), "rb");
            checkFileOpen(bottomMlpGradRefFp, bottomMlpGradRefFile);

            result = fread(dataInstance->hostInput().get(),
                           sizeof(DataT),
                           std::get<0>(problem.bwdDataSize),
                           inputFp);
            checkFileSize(inputFile, result, std::get<0>(problem.bwdDataSize));
            result = fread(dataInstance->hostUpstreamGrad().get(),
                           sizeof(DataT),
                           std::get<1>(problem.bwdDataSize),
                           upstreamGradFp);
            checkFileSize(upstreamGradFile, result, std::get<1>(problem.bwdDataSize));
            result = fread(dataInstance->hostGradRef().get(),
                           sizeof(DataT),
                           std::get<3>(problem.bwdDataSize),
                           gradRefFp);
            checkFileSize(gradRefFile, result, std::get<3>(problem.bwdDataSize));
            result = fread(dataInstance->hostBottomMlpGradRef().get(),
                           sizeof(DataT),
                           std::get<5>(problem.bwdDataSize),
                           bottomMlpGradRefFp);
            checkFileSize(bottomMlpGradRefFile, result, std::get<5>(problem.bwdDataSize));

            dataInstance->copyHostToDeviceBwdAll();
        }
    }
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::exec()
{
    if(mRunFlag)
    {
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

            // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
            const uint smem_rows_per_warp      = M_BLOCKS << 4;
            const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;

            const uint smem_elems_per_warp_acc
                = M_BLOCKS * kTileDim * SMEM_STRIDE_ACC * 2; // output in FP32
            const uint smem_elems_per_warp = (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
                                                 ? smem_elems_per_warp_mat
                                                 : smem_elems_per_warp_acc;
            uint       output_size         = NUM_COLS + (NUM_ROWS * (NUM_ROWS - 1) >> 1) + PAD;

            bool float4_predicate = !(output_size & 7); // (NUM_COLS & 7) || (output_size & 7));

            if(float4_predicate)
            {
                auto dlrmKernel = [this,
                                   num_rows_after_padding,
                                   num_cols_after_padding,
                                   smem_elems_per_warp,
                                   smem_rows_per_warp,
                                   output_size,
                                   num_row_steps,
                                   num_col_steps]() {
                    auto& dataInstance = DataStorage::instance();
                    hipExtLaunchKernelGGL((this->kernelFwdImpl()),
                                          (this->gridDim()),
                                          (this->blockDim()),
                                          (this->ldsUsage()),
                                          0,
                                          nullptr,
                                          nullptr,
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
                                          num_row_steps,
                                          num_col_steps,
                                          PAD);
                };

                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    dlrmKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                // Calculate efficiency
                auto& deviceInfo             = DeviceInfo::instance();
                auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<DataT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);
                mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }
            else
            {
                auto dlrmKernel = [this,
                                   num_rows_after_padding,
                                   num_cols_after_padding,
                                   smem_elems_per_warp,
                                   smem_rows_per_warp,
                                   output_size,
                                   num_row_steps,
                                   num_col_steps]() {
                    auto& dataInstance = DataStorage::instance();
                    hipExtLaunchKernelGGL((this->kernelFwdNonAlignedImpl()),
                                          (this->gridDim()),
                                          (this->blockDim()),
                                          (this->ldsUsage()),
                                          0,
                                          nullptr,
                                          nullptr,
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
                                          num_row_steps,
                                          num_col_steps,
                                          PAD);
                };

                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    dlrmKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                // Calculate efficiency
                auto& deviceInfo             = DeviceInfo::instance();
                auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<DataT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);
                mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }
        }
        else
        {
            const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;

            const uint TileSizeLog2 = Log2<TileSize>::value;

            uint input_dbytes = sizeof(DataT);

            uint row_tiles_per_step
                = NUM_ROWS > TileSize ? (TileSize == 32 ? 1 : kRowTilesPerStep) : 1;

            // num tiles
            uint num_row_tiles = (NUM_ROWS + TileSize - 1) >> TileSizeLog2;
            uint num_col_tiles = (NUM_COLS + TileSize - 1) >> TileSizeLog2;

            // number of rows and columns after padding
            uint num_rows_after_padding = num_row_tiles << TileSizeLog2;
            uint num_cols_after_padding = num_col_tiles << TileSizeLog2;

            // 2D ugrad size and stride
            uint interaction_ugrad_2D_stride = num_rows_after_padding + MEM_SKEW_SIZE;
            uint interaction_ugrad_2D_size_elems
                = num_rows_after_padding * interaction_ugrad_2D_stride;
            uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * input_dbytes;

            // 1D ugrad size
            uint interaction_ugrad_size              = NUM_ROWS * (NUM_ROWS - 1) >> 1;
            uint interaction_ugrad_size_with_padding = interaction_ugrad_size + PAD;

            // in_out place size and stride
            uint input_stride     = num_cols_after_padding + MEM_SKEW_SIZE;
            uint input_size_elems = num_rows_after_padding * input_stride;
            uint input_size_bytes = input_size_elems * input_dbytes;

            // sample size
            uint sample_size = NUM_ROWS * NUM_COLS;

            // output size
            uint output_size_elems
                = TileSize * TileSize * (TileSize == 32 ? 1 : kRowTilesPerStep) * kColTilesPerStep;
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

            bool float4_predicate
                = !((interaction_ugrad_size_with_padding & 7)); // || (NUM_COLS & 7));

            if(float4_predicate)
            {
                auto dlrmKernel = [this,
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
                                   shared_mem_per_warp_size_byte]() {
                    auto& dataInstance = DataStorage::instance();
                    hipExtLaunchKernelGGL((this->kernelBwdImpl()),
                                          (this->gridDim()),
                                          (this->blockDim()),
                                          (this->ldsUsage()),
                                          0,
                                          nullptr,
                                          nullptr,
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
                };

                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    dlrmKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                // Calculate efficiency
                auto& deviceInfo             = DeviceInfo::instance();
                auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<DataT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);
                mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }
            else
            {
                auto dlrmKernel = [this,
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
                                   shared_mem_per_warp_size_byte]() {
                    auto& dataInstance = DataStorage::instance();
                    hipExtLaunchKernelGGL((this->kernelBwdNonAlignedImpl()),
                                          (this->gridDim()),
                                          (this->blockDim()),
                                          (this->ldsUsage()),
                                          0,
                                          nullptr,
                                          nullptr,
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
                };

                hipEvent_t startEvent, stopEvent;
                CHECK_HIP_ERROR(hipEventCreate(&startEvent));
                CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

                CHECK_HIP_ERROR(hipEventRecord(startEvent));
                for(uint32_t i = 0; i < mRepeats; ++i)
                {
                    dlrmKernel();
                }
                CHECK_HIP_ERROR(hipEventRecord(stopEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

                auto timeMs = 0.0f;
                CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

                // Calculate efficiency
                auto& deviceInfo             = DeviceInfo::instance();
                auto  devicePeakGFlopsPerSec = deviceInfo->peakGFlopsPerSec<DataT>();

                mElapsedTimeMs        = float64_t(timeMs);
                mTotalGFlops          = calculateGFlops(mM, mN, mK);
                mMeasuredGFlopsPerSec = calculateGFlopsPerSec(mM, mN, mK, mElapsedTimeMs)
                                        * static_cast<float64_t>(mRepeats);
                mEfficiency = mMeasuredGFlopsPerSec / devicePeakGFlopsPerSec * 100.0;

                CHECK_HIP_ERROR(hipEventDestroy(startEvent));
                CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
            }
        }
    }
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::validateResults()
{
#ifdef WMMA_VALIDATION_TESTS
    if(mRunFlag)
    {
        auto& dataInstance = DataStorage::instance();
        if(!isBwd)
        {
            mValidationResult
                = allclose<DataT>(dataInstance->deviceOutputRef().get(),
                                  dataInstance->deviceOutput().get(),
                                  std::get<2>(dataInstance->currentDataSizeFwd()) * sizeof(DataT),
                                  false);
            EXPECT_TRUE(mValidationResult.pass);
        }
        else
        {
            mValidationResult
                = allclose<DataT>(dataInstance->deviceGradRef().get(),
                                  dataInstance->deviceGrad().get(),
                                  std::get<3>(dataInstance->currentDataSizeBwd()) * sizeof(DataT),
                                  false);
            EXPECT_TRUE(mValidationResult.pass);
            mValidationResult
                = allclose<DataT>(dataInstance->deviceBottomMlpGradRef().get(),
                                  dataInstance->deviceBottomMlpGrad().get(),
                                  std::get<5>(dataInstance->currentDataSizeBwd()) * sizeof(DataT),
                                  false);
            EXPECT_TRUE(mValidationResult.pass);
        }
    }
#endif
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::reportResults()
{
    if(!KernelI::sHeaderPrinted)
    {
        printHeader();
        KernelI::sHeaderPrinted = true;
    }

    printKernel();
}

template <uint32_t TileSize, typename DataT>
void DlrmKernelBase<TileSize, DataT>::tearDown()
{
}

#endif // DLRM_KERNEL_BASE_IMPL_H
