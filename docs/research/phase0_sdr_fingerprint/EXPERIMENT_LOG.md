# SOL SDR GPU EM 指纹实验记录

日期: 2026-05-14
设备: RSP1 (MSI2500+MSI001, 1df7:2500) 简易版
天线: 大环磁天线 (中波/长波), 0-30MHz 端口
SDR工具: miri_sdr (Zero-IF: -f 0 -i 0)

## 采集文件

| 文件 | 硬件 | 模式 | 负载 | 大小 |
|------|------|------|------|------|
| n14_dark_load.iq | 50Ω Load | 9.375M IF | — | 53M |
| n14_open_baseline.iq | 开路 | 9.375M IF | — | 53M |
| n14_gpu_signal.iq | 双GPU天线 | 9.375M IF | 空载 | 80M |
| n14_gtx1060.iq | GTX1060天线 | 9.375M IF | 空载 | 63M |
| n14_gtx1060_baseband.iq | GTX1060天线 | Zero-IF | 空载 | 80M |
| n14_gtx1060_loop_ant.iq | GTX1060磁环 | Zero-IF | 空载 | 107M |
| n14_gtx1060_load_loop.iq | GTX1060磁环 | Zero-IF | PyTorch matmul | 96M |

## 关键发现

### GTX 1060 (Pascal GP106)
- VRM 基频: 37.2 Hz (空载DCM)
- SMPS 开关: 50.9 kHz
- 负载下VRM跳变: 37.2→195.9 Hz (DCM→CCM)
- SMPS信号负载下 SNR 翻倍 (14.7x→26.5x)

### gfx1200 (RDNA4 9060XT) — 待采集
- 需单独安装后 Zero-IF 采集
- 预期VRM基频: ~109 Hz (基于之前9.375M IF测量)

### 互调产物
- 34.7, 176.0, 386.8 Hz 是双GPU共享电源的互调拍频，非单一GPU信号

## 脚本

- sol_capture.sh: 自动化SOL采集
- sol_analyze.py: SOL差分 + 频率分离
- full_scan_analyze.py: 全频段多分辨率分析
