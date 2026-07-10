# 开发过程记录

> ThreeChainMamba 项目开发过程 Issue 记录
> 本文档记录项目从架构设计到大规模实验的 8 个关键开发阶段，将在 Git 初始化后作为 Issue 创建。

---

## 目录

- [Issue #1: 三链 DNA-Mamba 架构设计与实现](#issue-1-三链-dna-mamba-架构设计与实现)
- [Issue #2: Mamba3 dt-RoPE 复数状态集成](#issue-2-mamba3-dt-rope-复数状态集成)
- [Issue #3: BP v1 alpha 门控死锁问题发现与修复](#issue-3-bp-v1-alpha-门控死锁问题发现与修复)
- [Issue #4: BP v2 乘法梯度死锁问题发现](#issue-4-bp-v2-乘法梯度死锁问题发现)
- [Issue #5: BP v2.1 加法独立调制修复](#issue-5-bp-v21-加法独立调制修复)
- [Issue #6: C500 国产 GPU 适配（__pycache__ 缓存陷阱）](#issue-6-c500-国产-gpu-适配__pycache__缓存陷阱)
- [Issue #7: 32 场景大规模实验设计与断点续跑](#issue-7-32-场景大规模实验设计与断点续跑)
- [Issue #8: 实验结果分析与性能报告](#issue-8-实验结果分析与性能报告)

---

## Issue #1: 三链 DNA-Mamba 架构设计与实现

### 问题描述

长程时空外推（long-horizon spatiotemporal extrapolation）是世界模型的核心能力。模型在训练时观察 T_train 步序列，推理时需对 T_eval >> T_train 的未来做预测。Transformer 受注意力二次复杂度与位置编码外推能力限制，难以在 O(N) 线性复杂度下实现稳定的长程外推。旧版 ThreeChain 架构存在六个失败模式（全零退化、位置编码越界、seed 方差为零假象等），需要重新设计架构。

### 解决方案

设计 ThreeChainMamba2 三链异构架构：在统一时空张量 `x:(B, T, N², d)` 上以三条异构 Mamba2 (SSD) 链做并行扫描，每层残差融合。

三条链的分工（类比世界树的根/干/枝）：

- **空间链 h_s**（双向 Mamba2，行+列扫描）：展开世界结构（树干），捕捉空间布局
- **时间链 h_t**（因果 Mamba2，沿 T 扫描）：深入时间长河（树根），捕捉时序演化
- **因果链 h_c**（K 维扫描）：分叉出多种未来（枝叶），捕捉 agent 间因果关系

配合 AnchorInit2 结构性锚定（x = cell_emb(S_0) + act_mean(actions) + time_emb(t)），解决旧架构全零退化问题。Cell 类型编码升级为 C=16（空格 / 智能体 ID 1-12 / 物品 ID 13-15），难度从 cell 变化率 6% 提升至 25%+。

### 关键文件

- `models/three_chain_mamba2.py` - 三链 Mamba2 核心架构（HeteroMamba2 + ThreeChainMamba2 + AnchorInit2）
- `models/common.py` - 共享组件（AnchorInit / Bind / Eagle / Fusion / 辅助损失头）
- `data/gen_grid_world.py` - GridWorld 数据生成器
- `data/dataset.py` - 数据集加载器（含 GroupedByNSampler / shuffle_labels sanity）
- `configs/default.yaml` - 默认训练配置
- `docs/architecture.md` - 架构详解与数学方程文档

### 状态

已关闭

---

## Issue #2: Mamba3 dt-RoPE 复数状态集成

### 问题描述

Mamba2 版本使用固定的 `nn.Embedding(max_T, d_model)` 时间嵌入，这是 OOD 长程外推衰减的罪魁祸首：当 T > max_T 时要么越界要么使用未训练的随机向量，还会干扰 Mamba 内部的位置信号。需要升级到 Mamba3 以获得更好的长度外推能力。

### 解决方案

集成官方 Mamba3（mamba_ssm 2.3.x），实现 ThreeChainMamba3，核心升级四点：

1. **移除固定长度 time_embed**：删除 Mamba2 时代遗留的 `nn.Embedding(max_T, d_model)`，消除外推衰减根源
2. **完全依赖 Mamba3 内核的 dt-RoPE**：官方 Mamba3 内置旋转位置编码基于时间步 dt 自适应累积角度 `cumulative_angles = cumsum(angle_raw * dt)`，不需要预设 max_len，天然支持任意长度外推
3. **复数状态空间**：通过 RoPE 旋转 B/C（key/query）投影，SSM 状态 h 获得等效复值结构，可追踪旋转/振荡/周期模式，解决实数状态长程遗忘问题
4. **梯形离散化**：trapezoidal rule 替代 Mamba2 的 ZOH，大步长下离散化误差更小

三链参数表（与 Mamba2 版对齐，仅去 d_conv）：

| 链 | d_state | expand | headdim | nheads | 方向 |
|----|---------|--------|---------|--------|------|
| 空间链 | 128 | 1 | 32 | 8 | 双向 |
| 时间链 | 64 | 2 | 64 | 8 | 因果 |
| 因果链 | 32 | 2 | 64 | 8 | 因果 |

提供 fallback 机制：当官方 mamba_ssm 2.3.x 不可用时，自动回退到 `mamba3_ref.py` 纯 Python 参考实现（本地验证用）。

### 关键文件

- `models/three_chain_mamba3.py` - 三链 Mamba3 架构（含 PairwiseBasePairMamba3）
- `models/mamba3_ref.py` - Mamba3 纯 Python 参考实现（fallback）
- `_mamba3_ref/mamba3.py` - Mamba3 参考实现研究笔记
- `verify_mamba3_rope.py` - dt-RoPE 验证脚本
- `docs/theory.md` - 三链不退化理论推导

### 状态

已关闭

---

## Issue #3: BP v1 alpha 门控死锁问题发现与修复

### 问题描述

BP v1（链内碱基对）使用 alpha 标量门控整个调制项，公式为 `x_s_new = x_s + alpha * (gamma*delta*x_s + beta - x_s)`。由于 alpha 初始为 0，导致梯度信号极弱：100 步训练后 alpha 均值仅 0.0007，调制网络几乎不更新。此外公式含 `-x_s` 项，即使 alpha > 0 但 gamma/beta = 0 时，输出 = `x_s * (1 - alpha)` 会缩小原始信号，非中性变换。

### 解决方案

设计 BP v2 跨链碱基对，从根本上重构调制方式：

- **去掉 alpha 门控**：改纯残差形式 `x_s_new = x_s + gamma*delta*x_s + beta`，调制网络末层零初始化 -> 初始 gamma=0, beta=0 -> 输出 = x_s（等价 baseline）
- **不含 -x_s 项**：纯残差不会缩小原始信号，保证中性初始化
- **跨链耦合**：碱基对加在 3 链之间（CrossChainBasePair 拉力），而非 v1 的链内（ComplementGate 通道互补）

v1 实验结果确认问题：BPv1（链内碱基对）OOD 下降 5.6%，给一条链戴枷锁反而损害外推能力。

### 关键文件

- `models/three_chain_mamba2_bp.py` - BP v1 链内碱基对（ComplementGate + AntiSymmetricHead + diversity_reg）
- `models/three_chain_mamba2_bpv2.py` - BP v2 跨链碱基对（CrossChainBasePair）
- `_check_bpv2_alpha.py` - BP v1 alpha 死锁诊断脚本
- `_bp_2000_log.txt` - BP v1 训练日志（alpha 死锁证据）

### 状态

已关闭

---

## Issue #4: BP v2 乘法梯度死锁问题发现

### 问题描述

BP v2 虽然解决了 v1 的 alpha 死锁，但在 C500 实战中发现了新的乘法梯度死锁问题。BP v2 时间链调制公式为 `x_t_new = x_t + reset_b * sgate_b * x_t`（乘法耦合），其中 `reset_b` 和 `sgate_b` 分别来自两个零初始化网络（`tc_c2t_reset` 和 `st_s2t_gate`）。

由于两个零初始化网络的输出相乘，根据链式法则，每个网络的梯度都包含对方的输出（=0）作为乘法因子，导致梯度互相阻塞 -> reset 和 sgate 永久死锁。C500 实测两者权重范数均为 0.0000。

### 解决方案

发现并确认问题，为 Issue #5 的 v2.1 加法修复提供诊断依据。通过 `_local_grad_test.py` 本地梯度测试复现并验证死锁现象：v2 乘法实现中 `bp1_gate` 和 `bp2_reset` 的梯度范数恒为 0.0000。

### 关键文件

- `models/three_chain_mamba3.py` - PairwiseBasePairMamba3 类（v2 乘法实现，第 262-263 行注释记录死锁问题）
- `_local_grad_test.py` - 梯度验证脚本（同时测试 v2 乘法和 v2.1 加法，对照对比）
- `_bpv2_2000_log.txt` - BP v2 训练日志（乘法死锁证据）
- `docs/call_logs.md` - C500 实战日志（bp1_gate=0.0000, bp2_reset=0.0000）

### 状态

已关闭

---

## Issue #5: BP v2.1 加法独立调制修复

### 问题描述

BP v2 的乘法耦合 `reset_b * sgate_b * x_t` 导致两个零初始化调制网络梯度互相阻塞，永久死锁。需要修改调制公式，使两个调制网络的梯度能独立流过。

### 解决方案

将时间链调制公式从乘法改为加法：

```
v2（乘法，死锁）:   x_t_new = x_t + reset_b * sgate_b * x_t
                     ↑ 两个零初始化网络相乘, 梯度链式法则含对方项(=0), 互相阻塞

v2.1（加法，修复）:  x_t_new = x_t + reset_b * x_t + sgate_b * x_t
                     ↑ 两个项独立加到 x_t 上, 各自梯度只依赖自身, 独立可流过
```

加法修复的关键性质：

1. **梯度独立**：`reset_b` 的梯度只含 `x_t` 项，`sgate_b` 的梯度只含 `x_t` 项，两者互不依赖
2. **初始等价 baseline**：零初始化时 `reset_b = sgate_b = 0`，输出 = `x_t + 0 + 0 = x_t`
3. **训练后双门控生效**：两种门控调制独立学习，互不阻塞

验证方式：

- **本地验证**（`_local_grad_test.py`）：v2.1 的 bp1_gate 和 bp2_reset 梯度范数非零，v2 的恒为 0
- **C500 实战验证**（`battle32_c500_log.txt`）：三 seed 的 bp1_gate=0.37~0.43, bp2_reset=0.37~0.42, bp3_gamma=0.39~0.47，全部非零

### 关键文件

- `models/three_chain_mamba3.py` - PairwiseBasePairMamba3 类（v2.1 加法实现，第 261-269 行）
- `_local_grad_test.py` - BP v2.1 梯度验证脚本（对比 v2 乘法 vs v2.1 加法）
- `_bp_v21_analysis.py` - BP v2.1 分析脚本
- `_bp_v21_final_report.py` - BP v2.1 最终报告生成脚本
- `docs/call_logs.md` - C500 实战日志（修复验证数据）
- `docs/demo_guide.md` - Demo 2 梯度验证说明

### 状态

已关闭

---

## Issue #6: C500 国产 GPU 适配（__pycache__ 缓存陷阱）

### 问题描述

在沐曦 MetaX C500 国产 GPU 上部署实验时遇到多个适配问题：

1. **__pycache__ 缓存陷阱**：Python 字节码缓存导致代码更新后仍运行旧版本，调试时产生难以定位的"幽灵 bug"
2. **rq_qos_wait 内核死锁**：C500（MXMACA）环境下 kernel 队列堆积导致 GPU 死锁，训练过程中随机挂起
3. **OOM 风险**：C500 显存管理与 NVIDIA 不同，BATCH_SIZE > 1 时容易 OOM
4. **mamba_ssm 版本兼容**：C500 上 mamba_ssm 2.2.4（Mamba2 baseline）与 2.3.x（Mamba3）需共存

### 解决方案

1. **__pycache__ 清理**：部署脚本中增加 `find . -type d -name __pycache__ -exec rm -rf {} +` 清理步骤，每次上传后强制清除缓存
2. **死锁防护五重策略**：
   - 每步训练后 `torch.cuda.synchronize()` 防 kernel 队列堆积
   - 模型间 `sleep(0.3s)` 释放 GPU 资源
   - BATCH_SIZE=1 防 OOM
   - 崩溃自动重试（最多 10 次），断点续跑不清除已有结果
   - 连续错误后 `mx-smi -r` GPU 重置
3. **环境变量隔离**：通过 `MAMBA3_FORCE_REF=1`、`MAMBA3_FORCE_GPU_SCAN=1`、`MAMBA3_USE_TRITON_SCAN=0` 控制 Mamba3 后端行为，纯 Python GPU scan 在 C500 上最优
4. **SSH 部署工具**：开发 `c500_plink.py`，通过环境变量配置连接参数（不硬编码），支持 upload / check / launch / progress / download 五个子命令

### 关键文件

- `c500_plink.py` - C500 SSH 部署工具（环境变量配置，安全推送）
- `battle32_c500_gpu.py` - 32 场景实验主脚本（加固版，含 synchronize 防护）
- `c500_run_battle32_v2.sh` - 实验启动脚本（含死锁防护 + 断点续跑 + 自动重试）
- `c500_smoke.py` - 环境冒烟测试（验证 mamba_ssm + BP 模块可用）
- `c500_auto_monitor.py` - 自动监控脚本
- `docs/mxmaca_adaptation.md` - MXMACA 国产 GPU 适配方案文档

### 状态

已关闭

---

## Issue #7: 32 场景大规模实验设计与断点续跑

### 问题描述

单一场景实验不足以全面验证三链架构的鲁棒性。需要设计覆盖多种网格规模、智能体密度、场景类型的大规模对比实验，同时解决 C500 上长时间运行（数小时）的可靠性问题：进程崩溃、GPU 死锁、网络中断等导致的实验中断。

### 解决方案

设计 32 场景 × 4 模型 × 3 seed = 384 组大规模实验：

**32 场景设计**（覆盖三类场景 × 多种配置）：

| 类别 | 场景数 | 网格 N | 智能体 K | 转移概率 p | 说明 |
|------|--------|--------|----------|------------|------|
| random | 8 | 6/8/12 | 4/8/12 | 0.0~0.6 | 完全随机动作 |
| goal_directed | 12 | 6/8/12 | 4/8/12 | 0.0~0.6 | 目标导向贪心路径 |
| adversarial | 8 | 6/8/12 | 4/8/12 | 0.0~0.6 | 对抗追踪 |
| extreme | 4 | 6/8/12 | 4/12 | 0.0~0.6 | 极端配置（最小最快/最大最密等） |

**4 模型对比**：

| 模型 | 说明 | 链数 |
|------|------|------|
| Transformer | 同参数基线 | - |
| Mamba3单链 | 单链消融 | 1 |
| 三链Mamba3 | 三链主力 | 3 |
| 三链Mamba3+BP | 三链 + 碱基对 | 3 + BP |

**断点续跑机制**：

- 启动时检查已有结果数（`NRES`），不删除已完成的实验
- 单模型失败不中断整体实验（异常捕获 + 继续下一个）
- 崩溃后自动重试（最多 10 次），每次重试前清理僵尸进程 + GPU 重置
- 实验结果实时写入 `battle32_results.json`，支持中断后恢复

**实验配置**：D_MODEL=64, BATCH_SIZE=1, MAX_STEPS=100, 3 seeds (42/123/456)

### 关键文件

- `battle32_c500_gpu.py` - 32 场景实验主脚本（含 SCENARIOS_32 定义 + 断点续跑）
- `c500_run_battle32_v2.sh` - 启动脚本（含重试循环 + 进程清理 + GPU 重置）
- `c500_plink.py` - 部署工具（upload / launch / progress / download）
- `battle32_c500_results.json` - 实验结果（384 组）
- `battle32_c500_stats.json` - 统计汇总
- `docs/call_logs.md` - 实验运行日志

### 状态

已关闭

---

## Issue #8: 实验结果分析与性能报告

### 问题描述

384 组大规模实验完成后，需要对结果进行系统性统计分析，生成可追溯的性能报告与可视化图表，验证三链架构在多场景下的长程外推不退化能力，并对比四类模型的差异化表现。

### 解决方案

开发统计分析与图表生成工具，生成完整性能报告：

**统计分析**（`stats_analysis.py`）：

- 各模型跨 32 场景 × 3 seed 的 val_ch / ood_ch / decay 均值与标准差
- 场景间稳定性分析（哪些场景三链优势最显著）
- 模型间显著性对比（三链 vs Transformer vs 单链）

**图表生成**（`regen_figures.py` / `generate_figures.py`）：

| 图表 | 内容 |
|------|------|
| fig1_training_dynamics.png | 训练动态曲线（loss + changed_acc vs step） |
| fig2_ood_curve.png | OOD 长程外推主图（changed_acc vs t，5 seed 均值±标准差） |
| fig3_ood_decay.png | OOD 衰减柱状图（越接近 0 越好） |
| fig4_param_efficiency.png | 参数效率散点图（模型大小 vs OOD 性能） |
| fig5_collapse_diagnosis.png | 退化诊断（zero_ratio + seed 方差） |
| fig6_seed_variance.png | Seed 方差箱线图（扁平=BUG，分散=真实学习） |
| fig7_ablation.png | 消融实验（BPv1/BPv2/Transformer vs baseline） |

**关键发现**：

1. 三链 Mamba3 OOD decay 在 ±2% 阈值内，长程外推不退化
2. 同参数 Transformer 在 OOD 区完全塌缩（zero_ratio=1.0，全猜 0）
3. BP v2.1 加法修复后，bp1_gate / bp2_reset / bp3_gamma 权重范数全部非零
4. 三链架构在 32 场景中均稳定，多场景鲁棒性验证通过

### 关键文件

- `stats_analysis.py` - 统计分析脚本
- `regen_figures.py` - 图表生成脚本（7 张图）
- `generate_figures.py` - 论文图表生成脚本
- `analyze_battle32_deep.py` - 32 场景深度分析
- `battle32_c500_stats.json` - 统计汇总结果
- `figures/` - 生成的图表目录
- `docs/performance_report.md` - 性能报告
- `PROFESSIONAL_REPORT.md` - 专业报告

### 状态

已关闭
