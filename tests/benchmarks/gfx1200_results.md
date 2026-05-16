# gfx1200 SWMMAC Benchmark Results

**GPU**: Radeon RX 9060 XT (RDNA4, gfx1200)
**CUs**: 32, **SIMDs**: 64, **Clock**: 2780 MHz (game), 2530 MHz (actual)
**Compiler**: LLVM 23 @ /opt/llvm-amd
**Date**: 2026-05-16

## Throughput (INT4, K=64, Dense)

| Kernel | TOPs | IPC | vs Baseline |
|--------|------|-----|-------------|
| K0 ChainPipeline (sync) | 769 | 0.132 | baseline |
| K6 StaggeredPipeline (reset) | 3582 | 0.614 | +366% |
| **K6 StaggeredPipeline (wrap)** | **4326** | **0.742** | **+462%** |

## 10-Minute Sustained (bench_sustained)

| Metric | Value |
|--------|-------|
| Mean | 3621 TOPs |
| Max | 4080 TOPs |
| Min | 2549 TOPs |
| StdDev | ±165 TOPs (4.6%) |
| Samples | 1,766,581 |

## Hardware Counters (rocprofv3)

| Counter | Value | Meaning |
|---------|-------|---------|
| VGPR | 136 | Close to estimate (139) |
| SQ_INSTS_VALU | 0 | SWMMAC on HWXDL, not VALU |
| Max Waves/SIMD | 16 | RDNA4 limit |
| SQ_BUSY_CYCLES | 1,243,174 | Non-XDL overhead ~19.9% |

## 26-Cycle Model

| Parameter | Value | Source |
|-----------|-------|--------|
| SWMMAC latency | 16 cyc | LLVM SISchedule.td |
| VGPR overhead | ~10 cyc | HW counter model |
| Effective pipeline | 26 cyc | Validated (P50 25.6μs vs 23.9μs) |
| Sync IPC | 0.615 | 16 waves / 26 cyc |
| L2 wrap IPC | 0.744 | ×1.21 boost factor |
| **Model error** | **+0.4%** | vs measured 4326 TOPs |

## SMPS EM Fingerprint (N14-locked, shielded)

| State | Frequency | Stability |
|------|-----------|-----------|
| Idle | 51049.9845 Hz | 0.0272 Hz (0.53 PPM) |
| Load | 51049.9874 Hz | 0.0261 Hz (0.51 PPM) |
| Δf | +0.0029 Hz | 0.06 PPM |

## Quantum Tunneling Factor (QTF)

| Band | Idle RMS | Load Clean RMS | Decrease |
|------|----------|----------------|----------|
| 0.1-10 Hz | 135.0 ns | 66.4 ns | -50.8% |
| 0.1-0.5 Hz | 35.2 ns | 11.4 ns | -67.6% |
| 1-5 Hz | 84.8 ns | 32.4 ns | -61.8% |
