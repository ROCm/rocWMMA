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
#ifndef ROCWMMA_FLOW_CONTROL_HPP
#define ROCWMMA_FLOW_CONTROL_HPP

namespace rocwmma
{

    namespace detail
    {
        // Perform synchronization across fragments(wavefronts) in a workgroup
        struct amdgcn_barrier
        {
            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_s_barrier();
            }
        };

        template <int32_t mask = 0>
        struct amdgcn_sched_barrier
        {
            enum : const int16_t
            {
                mask16 = mask
            };

            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_sched_barrier(mask16);
            }
        };

        template <int32_t priority = 0>
        struct amdgcn_setprio
        {
            enum : const int16_t
            {
                priority16 = priority
            };

            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_s_setprio(priority16);
            }
        };

        template <int32_t vmcnt = -1>
        struct amdgcn_s_waitcnt_vmcnt
        {
            enum : const int16_t
            {
                vmcnt16 = (vmcnt == -1) ? 0 : ((0xF) & vmcnt) & (((0x30) & vmcnt) << 10)
            };

            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_s_waitcnt(vmcnt);
            }
        };

        template <int32_t lgmcnt = -1>
        struct amdgcn_s_waitcnt_lgkmcnt
        {
            enum : const int16_t
            {
                lgmcnt16 = (lgmcnt == -1) ? 0 : (((0xF) & lgmcnt) << 8)
            };

            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_s_waitcnt(lgmcnt);
            }
        };

        template <int32_t vmcnt = -1, int32_t lgmcnt = -1>
        struct andgcn_s_waitcnt
        {
            enum : const int16_t
            {
                vmcnt16  = (vmcnt == -1) ? 0 : (((0xF) & vmcnt) & (((0x30) & vmcnt) << 10)),
                lgmcnt16 = (lgmcnt == -1) ? 0 : (((0xF) & lgmcnt) << 8),
                cnt      = vmcnt16 & lgmcnt16
            };

            __device__ static inline auto exec()
            {
                return __builtin_amdgcn_s_waitcnt(cnt);
            }
        };

    } // namespace detail

    using Barrier = detail::amdgcn_barrier;

    template <int32_t mask>
    using SchedBarrier = detail::amdgcn_sched_barrier<mask>;

    template <int32_t priority>
    using SetPrio = detail::amdgcn_setprio<priority>;

    template <int32_t vmcnt>
    using WaitcntVmcnt = detail::amdgcn_s_waitcnt_vmcnt<vmcnt>;

    template <int32_t lgmcnt>
    using WaitcntLgkmcnt = detail::amdgcn_s_waitcnt_lgkmcnt<lgmcnt>;

    template <int32_t vmcnt, int32_t lgmcnt>
    using Waitcnt = detail::andgcn_s_waitcnt<vmcnt, lgmcnt>;

} // namespace rocwmma

#endif // ROCWMMA_FLOW_CONTROL_HPP
