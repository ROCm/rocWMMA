# ROCm 上栈推进计划 — 从 SWMMAC 到 PyTorch 模型训练

## 优先级评估

| 目标 | 代码量 | 验证难度 | 影响面 | 快速可验证 | 优先级 |
|------|--------|---------|--------|-----------|--------|
| **composable_kernel** | 4724 files | ★★ | PyTorch FlashAttention, GEMM | ✅ 01_gemm 示例直接跑 | **P0** |
| Tensile | 1370 files | ★★★ | rocBLAS 80% GEMM | YAML 配置可快速测试 | P1 |
| Triton AMD | 345 files | ★★ | torch.compile 全模型 | 需改编译器 pass | P2 |
| RCCL | 96 files | ★★★★ | 多卡通信 | 需多 GPU + 网络 | P3 |

## P0: composable_kernel — 立刻可验证

### 现状
CK 提供 19 个 GEMM 示例 (`example/01_gemm` 到 `19_binary_elementwise`)，每个编译即跑。
`blockwise_gemm_xdl_traits.hpp` 是 XDL (SWMMAC) 矩阵核心的配置入口。
CK 使用 `ck_tile` 命名空间的现代 C++ 模板 API，与我们的 `StaggeredPipeline` 高度兼容。

### 三步验证计划

**Step 1: 替换 CK 的 GEMM 内核为 StaggeredPipeline**
- 文件: `example/01_gemm/` 
- 修改: 引用 `rocwmma/rocwmma_16chain.hpp`
- 替换: `ck_tile::BlockGemmXDL` → `rocwmma::StaggeredPipeline<16,1>`
- 验证: 编译运行，对比 TOPs

**Step 2: 在 CK 的 Block GEMM 层增加 atomicAdd 工作领取**
- 文件: `include/ck_tile/ops/gemm/block/block_gemm_xdl.hpp`
- 修改: 在 block 级 GEMM 入口增加 `atomicAdd` 工作领取逻辑
- 验证: CK 测试套件 `01_gemm` → `19_binary_elementwise`

**Step 3: 在 FlashAttention 融合算子中应用**
- 文件: CK 的 attention 实现 (GEMM + Softmax + GEMM)
- 应用: L2 持久化计数器 + StaggeredPipeline
- 验证: 对比标准 FlashAttention wall-clock 时间

### 预计收益
- 标准 GEMM: 与当前 CK GEMM 基准对比，预期 +10-20%
- FlashAttention: 消除 kernel 边界开销，预期 +15-30%

## P1: Tensile — YAML 驱动的 rocBLAS GEMM 代码生成

### 现状
Tensile 通过 YAML 配置文件穷举搜索最佳 GEMM 参数。
它在 rocBLAS 构建时生成汇编级 GEMM 内核。

### 验证路径
1. **最小侵入方案**: 在 `Tensile/Tensile/Configs/` 下新增一个 YAML 配置
   - 定义 `StaggeredWorkClaim=True` 选项
   - 当启用时，生成 atomicAdd-based work distribution
   - 回调到 rocWMMA 的 StaggeredPipeline

2. **快速测试**: 修改 `deep_bench_nn.csv` 添加我们的测试用例
   - 编译 rocBLAS with our Tensile config
   - 跑 rocBLAS GEMM benchmark 对比

### 预计收益
- rocBLAS GEMM: +5-15% (Tensile 本身已高度优化)
- 最大价值: 所有调用 rocBLAS 的框架自动受益

## P2: Triton AMD Backend — torch.compile 编译器层

### 现状
Triton 的 AMD 后端在 `third_party/amd/backend/compiler.py`。
它将 Triton IR 转换为 AMDGPU LLVM IR，控制 emit 策略。

### 验证路径
1. 在 compiler.py 的 `visit_LaunchOp` 或 kernel emit 阶段
   注入 `atomicAdd` work distribution 模式
2. 写一个简单的 Triton GEMM kernel 对比优化前后

### 预计收益
- 所有 `torch.compile` 模型自动获得原子散布优化
- 长期价值巨大，但需要频繁跟随 Triton 上游 rebase

## P3: RCCL — 多卡通信 (需要多 GPU)

### 验证路径
需要 2+ GPU 环境，当前单卡无法测试。
L2 持久化自旋锁可优化 AllReduce 同步。

---

## 立即执行: P0 Step 1

```bash
cd /home/yanli/work/ROCm/composable_kernel
# 查看现有 GEMM 示例
cat example/01_gemm/CMakeLists.txt
# 添加 rocWMMA include path
# 编译并运行基准对比
```
