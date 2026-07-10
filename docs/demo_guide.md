# Demo 演示说明文档

> ThreeChainMamba 三链 DNA-Mamba 项目演示指南
> 本文档面向评审与复现者，提供三个可独立运行的 Demo，覆盖本地验证、梯度修复验证与国产 GPU 大规模实验。

---

## 目录

1. [项目 Demo 概述](#1-项目-demo-概述)
2. [Demo 1: 本地快速运行](#2-demo-1-本地快速运行)
3. [Demo 2: BP v2.1 梯度验证](#3-demo-2-bp-v21-梯度验证)
4. [Demo 3: C500 国产 GPU 实验](#4-demo-3-c500-国产gpu实验)
5. [运行截图说明](#5-运行截图说明)
6. [演示视频说明](#6-演示视频说明)

---

## 1. 项目 Demo 概述

ThreeChainMamba 是基于 Mamba2/Mamba3 (SSD) 的三链状态空间模型，用于多智能体长程时空预测。核心创新在于三条异构链（空间链 / 时间链 / 因果链）并行扫描同一统一时空张量，配合 AnchorInit2 结构性锚定，实现长程外推不退化。

项目提供三个递进式 Demo：

| Demo | 名称 | 目的 | 运行环境 | 预计耗时 |
|------|------|------|----------|----------|
| Demo 1 | 本地快速运行 | GridWorld 数据生成 + 训练 + OOD 评估，验证长程外推不退化 | 本地 GPU (NVIDIA CUDA) | ~10 分钟 |
| Demo 2 | BP v2.1 梯度验证 | 验证碱基对加法修复是否解决乘法梯度死锁 | 本地 CPU/GPU 均可 | ~10 秒 |
| Demo 3 | C500 国产 GPU 实验 | 通过 SSH 部署并运行 32 场景 × 4 模型 × 3 seed 大规模实验 | 沐曦 MetaX C500 GPU 服务器 | ~8 小时 |

### 环境准备

所有 Demo 共用的基础环境：

```bash
# Python 3.10–3.12（3.14 可能缺少 mamba_ssm 预编译包）
pip install torch numpy pyyaml matplotlib
pip install triton mamba_ssm causal_conv1d
```

> 若 `mamba_ssm` 安装失败，请参考 [mamba_ssm 仓库](https://github.com/state-spaces/mamba) 匹配 CUDA/torch 版本。
> Demo 2（梯度验证）仅需 PyTorch，不依赖 mamba_ssm。

---

## 2. Demo 1: 本地快速运行

### 2.1 概述

本 Demo 在本地完成完整的「数据生成 → 训练 → OOD 评估」流程，使用 GridWorld 多智能体网格世界数据，验证三链模型在超出训练长度 50%（T=100 训练，T=150 评估）时的长程外推不退化能力。

### 2.2 步骤一：生成 GridWorld 训练数据

```bash
# 生成 GridWorld 训练数据（N=6/8/12, K=4/8/12, T=30/60/100）
python data/gen_grid_world.py
```

数据生成器（`data/gen_grid_world.py`）模拟 N×N 网格上 K 个智能体并行演化 T 步的过程：

- 动作集：上 / 下 / 左 / 右 / 停留（共 5 种）
- 冲突规则：同格多智能体 → 编号小者保留，其余回退
- 携带物：智能体可携带 token，冲突时按 `p_transfer` 概率转移
- Cell 类型编码：C=16（空格 / 智能体 ID 1-12 / 物品 ID 13-15）
- 三种场景：`random`（50% 随机动作）、`goal_directed`（30% 目标导向）、`adversarial`（20% 对抗追踪）

### 2.3 步骤二：训练三链模型

```bash
# 训练 ThreeChainMamba2（3.26M 参数），2000 步，seed=42
python train.py --config configs/matched_mamba2.yaml --model three_chain_mamba2 --seed 42

# 快速验证（100 步）
python train.py --config configs/default.yaml --model three_chain_mamba2 --seed 42 --max_steps 100
```

训练脚本（`train.py`）支持的模型列表：

| 模型 | 说明 |
|------|------|
| `three_chain_mamba2` | 三链 Mamba2 主力模型（baseline） |
| `three_chain_mamba3` | 三链 Mamba3（复数状态 + dt-RoPE） |
| `three_chain_mamba2_bp` | BP v1 链内碱基对（对照） |
| `three_chain_mamba2_bpv2` | BP v2 跨链碱基对（对照） |
| `transformer` | 同参数 Transformer（退化对照） |
| `single_chain` | 单链 Mamba（消融对照） |

训练过程输出示例：

```
step   0 | loss=1.5234 | changed_acc=0.08
step  50 | loss=0.8921 | changed_acc=0.35
step 100 | loss=0.6543 | changed_acc=0.52
...
step 2000 | loss=0.2103 | changed_acc=0.61
```

### 2.4 步骤三：OOD 长程外推评估

```bash
# OOD 评估：T=150（训练时 T=100），输出完整 t=1..150 曲线 + 衰减指标
python eval_ood.py --checkpoint results/best.pt --config configs/matched_mamba2.yaml \
    --data_root ./data/ood_T150 --out ood_metrics.json --seed 42
```

评估脚本（`eval_ood.py`）输出：

- `changed_acc_curve`：t=1..150 逐步准确率曲线
- `changed_acc`：OOD 区（t=101..150）平均准确率
- `ood_decay_100_to_150`：OOD 衰减（ch_acc@150 - ch_acc@100，越接近 0 越好）
- `zero_ratio`：全零预测比例（>0.90 视为退化塌缩）

### 2.5 预期结果

| 指标 | ThreeChainMamba2 | Transformer-tiny |
|------|------------------|-------------------|
| OOD changed_acc@150 | 0.5952 ± 0.0034 | 0.0000（塌缩） |
| OOD decay | -0.0042（≈0） | N/A |
| zero_ratio | 0.172（正常） | 1.000（全猜 0） |

> 同参数 Transformer 在 OOD 区完全塌缩成全猜 0，三链 SSM 稳定在 0.59，一眼可见「不退化」的核心优势。

### 2.6 非退化门检查

模型进入基准测试前须通过：

- `zero_ratio < 0.90`（不能全猜 0）
- `changed_acc > 0.30`（比随机 1/16 强）
- OOD 非零预测 > 目标的 5%

---

## 3. Demo 2: BP v2.1 梯度验证

### 3.1 概述

本 Demo 运行 `_local_grad_test.py`，验证碱基对（Base Pair）v2.1 加法修复是否成功解决了 v2 乘法梯度死锁问题。这是 BP 模块从 v1 → v2 → v2.1 迭代修复的关键验证步骤。

### 3.2 背景：BP 版本演进

| 版本 | 设计 | 问题 |
|------|------|------|
| BP v1 | 链内碱基对（ComplementGate 通道互补 + AntiSymmetricHead） | alpha 门控死锁：alpha 初始 0 → 梯度信号极弱 → 100 步后 alpha 均值仅 0.0007 |
| BP v2 | 跨链碱基对（去 alpha，纯残差，乘法耦合 `reset_b * sgate_b * x_t`） | 乘法梯度死锁：两个零初始化网络相乘 → 梯度互相阻塞 → reset 和 sgate 永久为 0 |
| **BP v2.1** | **加法独立调制（`reset_b * x_t + sgate_b * x_t`）** | **修复成功：两个项梯度独立可流过** |

### 3.3 运行验证

```bash
python _local_grad_test.py
```

### 3.4 预期输出

脚本同时测试 v2.1（加法）和 v2（乘法）两种实现，对比梯度：

```
=== v2.1 加法: 梯度检查 ===
  st_s2t_gate (bp1_gate): grad_norm = 0.001234 (应非零)
  tc_c2t_reset (bp2_reset): grad_norm = 0.000567 (应非零)

=== v2 乘法: 梯度检查 (对照) ===
  st_s2t_gate2 (bp1_gate): grad_norm = 0.000000 (死锁=0)
  tc_c2t_reset2 (bp2_reset): grad_norm = 0.000000 (死锁=0)

=== 结论 ===
v2.1 bp1_gate 梯度: 0.001234 (应非零)
v2.1 bp2_reset 梯度: 0.000567 (应非零)
v2   bp1_gate 梯度: 0.000000 (死锁=0)
v2   bp2_reset 梯度: 0.000000 (死锁=0)

✅ v2.1加法修复成功: bp1_gate和bp2_reset梯度非零, 不再死锁!
```

### 3.5 验证原理

核心差异在于时间链调制公式：

```
v2（乘法，死锁）:   x_t_new = x_t + reset_b * sgate_b * x_t
                     ↑ 两个零初始化网络相乘, 梯度链式法则含对方项(=0), 互相阻塞

v2.1（加法，修复）:  x_t_new = x_t + reset_b * x_t + sgate_b * x_t
                     ↑ 两个项独立加到 x_t 上, 各自梯度只依赖自身, 独立可流过
```

脚本还会测试完整 `PairwiseBasePairMamba3` 模块（若 mamba_ssm 可用），检查三对碱基对调制网络（bp1_gate / bp2_reset / bp3_gamma）的权重范数在一步 Adam 更新后的变化量。

---

## 4. Demo 3: C500 国产 GPU 实验

### 4.1 概述

本 Demo 通过 `c500_plink.py` 部署工具，远程连接沐曦 MetaX C500 国产 GPU 服务器，运行 32 场景 × 4 模型 × 3 seed = 384 组实验的大规模对比，验证三链架构在国产 GPU 上的可迁移性与多场景鲁棒性。

### 4.2 实验配置

| 配置项 | 值 |
|--------|-----|
| GPU | MetaX C500（沐曦 MXMACA） |
| 显存 | 64.0 GB |
| mamba_ssm | 2.2.4 |
| D_MODEL | 64 |
| BATCH_SIZE | 1（内存安全） |
| MAX_STEPS | 100 |
| 场景数 | 32（random/goal_directed/adversarial 三类） |
| 模型数 | 4（Transformer / Mamba3单链 / 三链Mamba3 / 三链Mamba3+BP） |
| Seeds | 3（42 / 123 / 456） |
| 总实验数 | 384 |
| 防护策略 | 每步 `cuda.synchronize()` + 模型间 `sleep(0.3s)` + BATCH_SIZE=1 |

### 4.3 连接配置

通过环境变量配置 SSH 连接（不硬编码在源码中，确保安全）：

```bash
# 方式1: 创建 .env 文件（推荐，.env 不会被提交到 git）
cat > .env << 'EOF'
C500_SSH_HOST=your.host.ip
C500_SSH_PORT=32222
C500_SSH_USER=your_username
C500_SSH_PASS=your_password
C500_SSH_HOSTKEY=SHA256:xxxxx
EOF

# 方式2: 直接设置环境变量（Windows PowerShell）
$env:C500_SSH_HOST="your.host.ip"
$env:C500_SSH_PORT="32222"
$env:C500_SSH_USER="your_username"
$env:C500_SSH_PASS="your_password"
$env:C500_SSH_HOSTKEY="SHA256:xxxxx"
```

### 4.4 运行步骤

```bash
# 步骤1: 上传所有文件到 C500
python c500_plink.py upload

# 步骤2: 检查 C500 环境（验证 mamba_ssm + BP 模块可用）
python c500_plink.py check

# 步骤3: 启动 32 场景实验（断点续跑，不清除已有结果）
python c500_plink.py launch

# 步骤4: 检查实验进度
python c500_plink.py progress

# 步骤5: 下载结果文件
python c500_plink.py download
```

也可一键执行全部步骤：

```bash
python c500_plink.py all
```

### 4.5 上传文件清单

`c500_plink.py upload` 上传以下文件到远程 `/root/three_chain_v3`：

- `battle32_c500_gpu.py` — 32 场景实验主脚本（加固版）
- `stats_analysis.py` — 统计分析脚本
- `c500_smoke.py` — 环境冒烟测试
- `c500_run_battle32_v2.sh` — 实验启动脚本（含死锁防护）
- `models/` 目录 — 三链 Mamba3 / Mamba2 / BP / baselines 全部模型代码
- `data/` 目录 — GridWorld 数据生成器与数据集加载器

### 4.6 32 场景设计

实验覆盖三类场景 × 多种网格规模与智能体密度：

| 类别 | 场景数 | 说明 |
|------|--------|------|
| random | 8 | 完全随机动作，含小/中/大网格 × 少/中/多 agent |
| goal_directed | 12 | 目标导向贪心路径，含强转移/长程变体 |
| adversarial | 8 | 对抗追踪（pursuer vs escaper），含高密度强转移 |
| extreme | 4 | 极端配置：最小最快 / 最大最密 / 高转移随机 / 无人机轨迹 |

### 4.7 死锁防护策略

C500（MXMACA）环境下存在 `rq_qos_wait` 内核死锁风险，实验脚本采取以下加固措施：

1. **每步 `torch.cuda.synchronize()`**：防止 kernel 队列堆积（不全局阻塞，只防溢出）
2. **崩溃自动重试**：最多 10 次重试，断点续跑不清除已有结果
3. **启动时清理僵尸进程**：`pkill -9` 清理残留 Python 进程
4. **BATCH_SIZE=1**：防 OOM
5. **连续错误后 GPU 重置**：`mx-smi -r` 重置 GPU 状态
6. **模型间 `sleep(0.3s)`**：释放 GPU 资源，防止连续模型训练死锁

### 4.8 下载结果

`c500_plink.py download` 下载以下文件到本地：

| 远程文件 | 本地文件 | 内容 |
|----------|----------|------|
| `battle32_results.json` | `battle32_c500_results.json` | 384 组实验完整结果 |
| `battle32_stats.json` | `battle32_c500_stats.json` | 统计分析汇总 |
| `battle32_c500_log.txt` | `battle32_c500_log.txt` | 完整运行日志 |

---

## 5. 运行截图说明

`figures/` 目录下包含 7 张关键图表，由 `regen_figures.py` 生成（宇宙深色风格）：

| 图表文件 | 内容说明 |
|----------|----------|
| `fig1_training_dynamics.png` | 训练动态曲线：loss + changed_acc 随训练步数变化（seed=42），对比四类模型 |
| `fig2_ood_curve.png` | **OOD 长程外推主图**：changed_acc vs 时间步 t（1-150），5 seed 均值±标准差，T=100 处标注 OOD 区起点 |
| `fig3_ood_decay.png` | OOD 衰减柱状图：ch_acc@150 - ch_acc@100（5 seed），越接近 0 越好 |
| `fig4_param_efficiency.png` | 参数效率散点图：模型参数量 vs OOD changed_acc，★ 标记 5 seed 均值 |
| `fig5_collapse_diagnosis.png` | 退化诊断：左图 zero_ratio 对比（旧 ThreeChain 97.7% 退化 vs 新 Mamba2 17.2% 正常），右图 5 seed 标准差（std=0 → BUG 假象） |
| `fig6_seed_variance.png` | Seed 方差箱线图：5 seed OOD 性能分布（扁平箱 = std=0 BUG，分散箱 = 真实学习） |
| `fig7_ablation.png` | 消融实验：BPv1（链内，OOD 降 5.6%）/ BPv2（跨链，持平）/ Transformer（塌缩）vs baseline |

### 重新生成图表

```bash
python regen_figures.py
# 输出: figures/fig1..fig7.png（宇宙深色风，ASCII 安全）
```

---

## 6. 演示视频说明

> **状态：待录制**

| 视频 | 内容 | 时长 | 链接 |
|------|------|------|------|
| Demo 1 演示 | 本地 GridWorld 数据生成 + 训练 + OOD 评估全流程 | ~5 分钟 | _待录制后填写_ |
| Demo 2 演示 | BP v2.1 梯度验证运行与结果解读 | ~2 分钟 | _待录制后填写_ |
| Demo 3 演示 | C500 国产 GPU 实验部署与进度监控 | ~5 分钟 | _待录制后填写_ |
| 项目总览 | 三链架构讲解 + 核心实验结果解读 | ~10 分钟 | _待录制后填写_ |

---

## 附录：关键文件索引

| 文件 | 作用 |
|------|------|
| `models/three_chain_mamba2.py` | 三链 Mamba2 核心架构（HeteroMamba2 + ThreeChainMamba2 + AnchorInit2） |
| `models/three_chain_mamba3.py` | 三链 Mamba3 架构（复数状态 + dt-RoPE + PairwiseBasePairMamba3） |
| `models/three_chain_mamba2_bp.py` | BP v1 链内碱基对（ComplementGate + AntiSymmetricHead） |
| `models/three_chain_mamba2_bpv2.py` | BP v2 跨链碱基对（CrossChainBasePair） |
| `train.py` | 训练入口 |
| `eval_ood.py` | OOD 长程外推评估 |
| `data/gen_grid_world.py` | GridWorld 数据生成器 |
| `_local_grad_test.py` | BP v2.1 梯度验证脚本 |
| `c500_plink.py` | C500 国产 GPU 部署工具 |
| `battle32_c500_gpu.py` | 32 场景实验主脚本（加固版） |
| `c500_run_battle32_v2.sh` | C500 实验启动脚本（含死锁防护） |
| `regen_figures.py` | 图表生成脚本 |
