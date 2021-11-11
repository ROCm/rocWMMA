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

#ifndef DLRM_KERNEL_BASE_H
#define DLRM_KERNEL_BASE_H

#include <iostream>
#include <sstream>
#include <string>

#include "Constants.h"
#include "DlrmResource.h"
#include "HipDevice.h"
#include "common.h"

// Basic structure to hold runtime problem
// parameters
struct ProblemParams
{
    std::pair<int64_t, int64_t>           threadBlockSize;
    std::tuple<int64_t, int64_t, int64_t> problemSize;
    // Input, Output, OutputRef
    std::tuple<int64_t, int64_t, int64_t> fwdDataSize;
    // Input, UpstreamGrad, Grad, GradRef, BottomMlpGrad, BottomMlpGradRef
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> bwdDataSize;
    bool                                                             isBwd;
};

// Defines for hard-coded template parameters
enum : uint32_t
{
    // Shared kernel template parameters
    kWarpSize     = AMDGCN_WAVE_SIZE,
    kWarpSizeLog2 = Log2<AMDGCN_WAVE_SIZE>::value,
    kTileDim      = 16,
    kTileDimLog2  = Log2<kTileDim>::value,

    // Forward kernel template parameters
    warps_per_threadblock = 128 / kWarpSize,
    threadblock_size      = warps_per_threadblock * kWarpSize,
    M_BLOCKS              = 2,
    K_BLOCKS              = 8,
    SMEM_STRIDE           = K_BLOCKS * 16 + 8,
    SMEM_STRIDE_ACC       = M_BLOCKS * 16 + 8,

    // Backwards kernel template parameters
    kWarpsPerBlock   = 128 / kWarpSize,
    kNumThreads      = kWarpsPerBlock * kWarpSize,
    kRowTilesPerStep = 32 / kTileDim,
    kColTilesPerStep = 1,

    // Data sizes
    BATCH_SIZE = 64,
    NUM_ROWS   = 27,
    NUM_COLS   = 128,
    PAD        = 0
};

// Typeless Kernel interface to use with testing harness.
struct KernelI
{
    KernelI() {}
    virtual ~KernelI(){};

    virtual void          setup(ProblemParams const& problem)                 = 0;
    virtual void          exec()                                              = 0;
    virtual void          validateResults()                                   = 0;
    virtual void          reportResults()                                     = 0;
    virtual void          tearDown()                                          = 0;
    virtual std::ostream& printHeader(std::ostream& stream = std::cout) const = 0;
    virtual std::ostream& printKernel(std::ostream& stream = std::cout) const = 0;

    static bool sHeaderPrinted;
};

inline std::ostream& operator<<(std::ostream& stream, KernelI const& kernel)
{
    kernel.printHeader(stream);
    kernel.printKernel(stream);
    return stream;
}

// Typed DLRM kernel that provides the basis for DLRM tests.
// This class provides common implementation code.
template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t TileSize, typename DataT>
struct DlrmKernelBase : public KernelI
{
protected: // Types
    // Shared access to DLRM storage
    using DataStorage = DlrmResource<DataT>;
    // Using Hip device backend
    using DeviceInfo = HipDevice;

    // Interface to forward device kernel
    using KernelFwdFunc = void (*)(const DataT* __restrict, // input
                                   DataT* __restrict, //  output
                                   uint32_t, // batch_size
                                   uint32_t, // num_rows
                                   uint32_t, // num_cols
                                   uint32_t, // num_rows_after_padding
                                   uint32_t, // num_cols_after_padding
                                   uint32_t, // smem_elems_per_warp
                                   uint32_t, // smem_rows_per_warp
                                   uint32_t, // output_size
                                   uint32_t, // num_row_steps
                                   uint32_t, // num_col_steps
                                   uint32_t); // pad

    // Interface to backwards device kernel
    using KernelBwdFunc = void (*)(const DataT* __restrict, // input
                                   const DataT* __restrict, // upstream_grad
                                   DataT* __restrict, // grad
                                   DataT* __restrict, // bottom_mlp_grad
                                   uint32_t, // batch_size
                                   uint32_t, // num_rows
                                   uint32_t, // num_cols
                                   uint32_t, // num_rows_after_padding
                                   uint32_t, // num_cols_after_padding
                                   uint32_t, // sample_size
                                   uint32_t, // interaction_ugrad_size
                                   uint32_t, // interaction_ugrad_size_with_padding
                                   uint32_t, // interaction_ugrad_2D_size_elems
                                   uint32_t, // interaction_ugrad_2D_stride
                                   uint32_t, // input_size_elems
                                   uint32_t, // input_stride
                                   uint32_t, // num_row_steps
                                   uint32_t, // num_col_steps
                                   uint32_t, // row_tiles_per_step
                                   uint32_t); //shared_mem_per_warp_size_byte

protected:
    DlrmKernelBase();
    virtual ~DlrmKernelBase();

    // Kernels MUST provide the device kernel function.
    virtual KernelFwdFunc kernelFwdImpl() const           = 0;
    virtual KernelFwdFunc kernelFwdNonAlignedImpl() const = 0;
    virtual KernelBwdFunc kernelBwdImpl() const           = 0;
    virtual KernelBwdFunc kernelBwdNonAlignedImpl() const = 0;

    // Kernel launch parameters
    virtual uint32_t ldsUsage() const;
    virtual dim3     gridDim() const;
    virtual dim3     blockDim() const;

    // Kernel run checks.
    // True = run test
    // False = skip test
    virtual bool checkDevice() const;
    virtual bool checkSizes() const;
    virtual bool checkLds() const;

    // Reset all members to default values
    virtual void reset();

public:
    // KernelI interface fulfillment
    virtual void          setup(ProblemParams const& problem) override;
    virtual void          exec() override;
    virtual void          validateResults() override;
    virtual void          reportResults() override;
    virtual void          tearDown() override;
    virtual std::ostream& printHeader(std::ostream& stream = std::cout) const override;
    virtual std::ostream& printKernel(std::ostream& stream = std::cout) const override;

protected:
    // Problem params for kernel
    uint32_t mTBlockX, mTBlockY;
    uint32_t mM, mN, mK;

    // Execution flow control
    uint32_t mRepeats;
    bool     mRunFlag          = true;
    bool     mValidationResult = false;
    bool     isBwd             = false;

    // Performance
    float64_t mTotalGFlops, mMeasuredGFlopsPerSec;
    float64_t mElapsedTimeMs, mEfficiency;
};

#include "DlrmKernelBase_impl.h"

#endif // DLRM_KERNEL_BASE_H
