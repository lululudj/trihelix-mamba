# ThreeChainMamba2 Wiki 首页

> 🌳 **想看世界树的分支吗？三螺旋带你见证未来。**
> *Care to witness Yggdrasil's branches? Three helices bear every future.*
>
> **三链 = 世界树的根/干/枝**: 时间链 $h_t$ 深入时间长河 (树根), 空间链 $h_s$ 展开世界结构 (树干), 因果链 $h_c$ 分叉出多种未来 (枝叶)。
> 同参数 Transformer 看不见枝叶 (zero_ratio=1.0 全猜 0), 三链在长程外推里看见每一个分支 (OOD decay ≈ 0)。

## 目录 (TOC)

1. [项目简介](#1-项目简介)
2. [文档索引](#2-文档索引)
3. [快速开始](#3-快速开始)
4. [实验数据索引](#4-实验数据索引)
5. [沐曦青年开源专项基金申报信息](#5-沐曦青年开源专项基金申报信息)
6. [联系与引用](#6-联系与引用)

---

## 1. 项目简介

**ThreeChainMamba2 (三链 DNA-Mamba2)** 是基于 Mamba2 (SSD) 的三链状态空间模型, 用于多智能体长程时空预测。核心创新:

> **三链 (空间 / 时间 / 因果) 异构扫描同一张量 + AnchorInit2 结构性锚定 → 100M → 1.13B 参数长程外推不退化**

| 项 | 内容 |
|---|---|
| 项目名 | ThreeChainMamba2 (三链 DNA-Mamba2) |
| 一句话简介 | 基于 Mamba2 (SSD) 的三链状态空间模型, 用于多智能体长程时空预测; 在国产沐曦 MXMACA GPU 上可零修改迁移 |
| 核心创新 | 三链 (空间 / 时间 / 因果) 异构扫描同一张量 + AnchorInit2 结构性锚定 → 100M → 1.13B 参数长程外推不退化 |
| 实验亮点 | GridWorld (参数扩展 30M → 1.13B) + SDD 真实数据 (bookstore 7 视频 + nexus 12 视频) 双场景验证 |
| 国产 GPU 落地 | `mamba_ssm` 基于 Triton, 沐曦 Triton-MXMACA 编译后端可零修改适配, 适合端侧人形机器人多智能体实时推演 |
| 开源许可 | MIT (OSI 认可) |

### 1.1 为什么重要

长程时空外推 (long-horizon spatiotemporal extrapolation) 是世界模型的核心能力: 模型在训练时观察 $T_{train}$ 步序列, 推理时需对 $T_{eval} \gg T_{train}$ 的未来做预测。Transformer 受注意力二次复杂度与位置编码外推能力限制; 近年 SSM (Mamba / Mamba2) 在线性复杂度下展示出长程建模潜力, 但其外推稳定性随规模放大是否保持, 此前缺乏严格验证。

本工作在 **$O(N)$ 线性复杂度** 下, 于 **[100M → 1.13B]** 参数区间验证长程外推不退化 (单点 decay 与窗口 decay 双双全部落在 ±2% 内, 1B 三 seed spread 仅 0.5%), 并经三个控制实验 (随机标签 / 多 seed / 步数对照) 严格证明这是真实泛化而非指标噪声。

### 1.2 核心结果速览

| 规模 | params | 步数 | seed 数 | 单点 decay (T150) | 窗口 decay (T150) |
|------|--------|------|--------|-------------------|-------------------|
| 100M | 72.32M | 500 | 3 | +0.24% / +1.93% / +0.37% | -0.79% / +0.67% / -0.15% |
| 300M | 182.95M | 2000 | 3 | -1.69% / -1.66% / -0.28% | -0.36% / -0.15% / -0.16% |
| 700M | 724.51M | 2000 | 2 | +0.19% / -0.39% | -0.02% / -0.60% |
| **1B (gradient_ckpt)** | **1.13B** | 2000 | 3 | +0.28% / -0.12% / +0.38% | -0.01% / -0.39% / -0.30% |

主结论: 100M → 1.13B 全部 converged 实验, 单点 decay 与窗口 decay **双双全部落在 ±2%** 内; 5× 外推 (T=500) 仍保持 (单点 decay = +0.36%); 大模型方差更小 (1B spread=0.5% < 100M 1.7%)。

---

## 2. 文档索引

本 Wiki 包含 4 个核心文档, 按阅读顺序排列:

### 2.1 架构详解 (推荐先读)

📄 **[architecture.md — 架构详解与数学方程](architecture.md)**

- 三链架构图 (ASCII art)
- 完整数学方程 (LaTeX 风格): AnchorInit2 / 三链扫描 / 残差融合 / 输出头
- Mamba2 (SSD) 核心方程: 状态递推 / SSD 分块 / heavy_tail_activation
- 三链参数表 (空间 / 时间 / 因果的 $d_{state}$ / $d_{conv}$ / $expand$ / $headdim$)
- 因果性分析 (时间因果 / 空间双向 / agent 因果)
- 实现硬约束

### 2.2 理论推导 (科研深度)

📄 **[theory.md — 三链不退化理论推导](theory.md)**

- OOD 长程外推退化现象定义
- 5 个可证伪假说 (H1-H5, 3 支持 / 2 推翻)
- 三层保障机制 (核心理论贡献): 基础层 (AnchorInit2) / 稳定层 (残差+LayerNorm) / 优化层 (三链 SSM)
- GridWorld vs SDD 差异理论 (任务复杂度函数)
- window-smoothed decay 理论 (单点噪声 + ±5 邻域均值稳健性)
- 与 Transformer / Mamba3 的机制对比

### 2.3 国产 GPU 适配 (沐曦基金申报核心)

📄 **[mxmaca_adaptation.md — MXMACA 国产 GPU 适配方案](mxmaca_adaptation.md)**

- MXMACA 软件栈架构图 (文字版)
- 零修改适配证明 (PyTorch 算子 / Triton 内核 / cu-bridge)
- 部署步骤 (沐曦 GPU 环境 / 安装 / 验证)
- 端侧推理 benchmark 计划 (曦思 N 系列延迟 / 显存预估)
- 落地场景: 人形机器人多智能体实时推演 (SSM $O(L)$ vs Transformer $O(L^2)$)
- 风险与缓解

### 2.4 项目主页 (本页)

📄 **[Home.md — Wiki 首页 (本页)](Home.md)**

---

## 3. 快速开始

### 3.1 环境要求

- Python ≥ 3.10 (3.14 可能缺 `mamba_ssm` 预编译 wheel)
- CUDA GPU (≥ 8GB 显存; 1B 训练需 gradient checkpointing + ≥ 24GB)
- 国产 GPU: 沐曦 MXMACA 软件栈 (提供 Triton 编译后端, 可零修改适配, 详见 [mxmaca_adaptation.md](mxmaca_adaptation.md))

### 3.2 安装

```bash
# 1. 基础依赖
pip install torch numpy pyyaml matplotlib

# 2. 安装 triton (Mamba2 kernel 依赖)
pip install triton

# 3. 安装 mamba_ssm 与 causal_conv1d
#    注意: 必须用 --no-build-isolation, 复用已安装的 torch CUDA 版本编译
pip install --no-build-isolation mamba_ssm causal_conv1d

# 或从源码编译 (沐曦 MXMACA 等非 CUDA 后端同理, Triton 后端会自动选择)
git clone https://github.com/state-spaces/mamba.git
cd mamba && MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --no-deps -e .
```

### 3.3 5 分钟复现主结果

```bash
# 1. 生成 GridWorld 数据
python data/gen_grid_world.py

# 2. 训练 (3.26M 主力模型)
python train.py --config configs/matched_mamba2.yaml --model three_chain_mamba2 --seed 42

# 3. OOD 长程外推评估 (--extend_max_T 扩展 time_embed 到未见长度)
python eval_ood.py --checkpoint results/best.pt --extend_max_T 1024

# 4. 退化诊断 (非退化门)
python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 2000 --seed 42

# 5. 同参数 Transformer 对照
python probe_mamba2_brain.py --model transformer --config configs/matched_transformer_tiny.yaml --max_steps 2000 --seed 42
```

### 3.4 大规模训练 (100M → 1.13B)

```bash
# 100M / 300M / 700M / 1B 对应配置
python train.py --config configs/matched_mamba2_100m.yaml  --seed 0
python train.py --config configs/matched_mamba2_300m.yaml  --seed 0
python train.py --config configs/matched_mamba2_700m.yaml  --seed 0
python train.py --config configs/matched_mamba2_1000m.yaml --seed 0   # 1B, 需 use_checkpoint

# SDD 真实数据
python data/gen_sdd_grid.py
python train.py --config configs/sdd_mamba2_30m.yaml          --max_steps 10000
python train.py --config configs/sdd_mamba2_30m_nexus.yaml    --max_steps 10000
```

详见项目根 [README.md](../README.md) 与 [REPRODUCE.md](../REPRODUCE.md)。

---

## 4. 实验数据索引

所有实验数据可追溯到 `results_cloud/` 与 `results_stage2/` 原始 JSON 文件。

### 4.1 主数据汇总

📊 **[paper/data_summary.md — 实验数据汇总表](../paper/data_summary.md)**

由 `summarize_all_experiments.py` 自动生成, 包含:

- **规模实验** (SSM 不退化): 30M / 100M / 300M / 700M / 1B / 1B (gradient_ckpt) 完整 decay 数据
- **HTA 移植**: heavy_tail_activation 从 Mamba3 移植到 Mamba2 的对照实验
- **超深模型** (n_layers=8): 105M deep_n8 在 T150 / T300 / T500 上的 decay
- **超长 OOD 外推**: T300 / T500 极端外推
- **gradient checkpointing**: 1B 训练显存从 25GB 降到 10.2GB
- **控制实验**: 随机标签 sanity / B@1000 vs B@500 / 1B 多 seed 统计

### 4.2 机制分析报告

📄 **[paper/mechanism_analysis.md — 长程外推不退化机制分析](../paper/mechanism_analysis.md)**

阶段 3 机制深挖最终报告 (3.1 探针 + 3.2 消融 + 3.3 对照), 5 个假说验证 (3 支持 / 2 推翻)。

### 4.3 消融实验报告

📄 **[paper/stage3_ablation.md — 阶段 3.2 消融实验报告](../paper/stage3_ablation.md)**

30M 模型三链消融 (置零某链输出, 保持参数量), 包含 baseline / no_spatial / no_temporal / no_causal / no_all 五种配置。

### 4.4 信号保留对照报告

📄 **[paper/stage3_3_retention.md — 阶段 3.3 SSM vs Transformer 信号保留对比](../paper/stage3_3_retention.md)**

H4 / H5 假说验证, 揭示 "信号保留 ≠ 预测质量" 的反直觉发现。

### 4.5 SDD 真实数据报告

📄 **[paper/stage2_sdd_report.md — Stage 2 SDD 真实数据验证报告](../paper/stage2_sdd_report.md)**

脱离 toy 网格世界, 在 Stanford Drone Dataset bookstore 7 视频上验证 SSM 不退化机制。

### 4.6 三链 norm 探针报告

📄 **[paper/stage3_3_1_validation.md — 阶段 3.1 Hidden State 探针假说验证报告](../paper/stage3_3_1_validation.md)**

H1 / H2 / H3 验证, 三链 norm 在 OOD 区稳定性 + 三链信息正交性发现。

### 4.7 完整技术报告

📄 **[paper/technical_report.md — 完整技术报告 v0.3](../paper/technical_report.md)**

含 arXiv 风格章节, LaTeX 源见 [paper/main.tex](../paper/main.tex) 与 [paper/sections/](../paper/sections/)。

---

## 5. 沐曦青年开源专项基金申报信息

### 5.1 项目契合沐曦基金方向

| 沐曦基金方向 | ThreeChainMamba2 对应 |
|---|---|
| 国产 GPU 适配 | MXMACA 软件栈零修改适配 (详见 [mxmaca_adaptation.md](mxmaca_adaptation.md)) |
| 开源社区贡献 | 全部代码 (models / train.py / eval_ood.py) + 实验数据 (results_stage2 / results_cloud) + 论文源码 (paper/) MIT 开源 |
| 端侧应用落地 | 端侧人形机器人多智能体实时推演 (SSM $O(L)$ 内存 vs Transformer $O(L^2)$) |
| 大模型训练 | 1.13B 参数模型经 gradient checkpointing 在 24GB GPU 上训练完成 |
| 算子层创新 | heavy_tail_activation 从 Mamba3 移植到 Mamba2 (不改系统包源码, parametrize 注入) |

### 5.2 适配准备度

| 项 | 状态 |
|---|---|
| PyTorch 算子适配 | ✅ 已完成 (Linear / LayerNorm / Embedding / GELU / Conv2d 全在 MXMACA 2650 算子集) |
| Triton 内核适配 | 🟡 计划阶段 4 (mamba_ssm 全部 Triton kernel 应可由 Triton-MXMACA 自动编译) |
| 端侧推理 benchmark | 🟡 计划阶段 4 (曦思 N 系列延迟 / 显存预估见 [mxmaca_adaptation.md §5](mxmaca_adaptation.md)) |
| 1B 大模型训练 | 🟡 计划阶段 4 (gradient checkpointing 在 MXMACA 上行为待验证) |

### 5.3 路线图 (与沐曦基金对齐)

- [x] **阶段 1** — 核心模型 + 5-seed 基准 + 消融 (3.26M)
- [x] **阶段 2** — 参数扩展 (100M → 1.13B) + SDD 真实数据验证
- [x] **阶段 3** — 机制深挖 (AnchorInit2 三层保障 + 5 假说验证) + 论文
- [ ] **阶段 4** — 沐曦 MXMACA GPU 实测 + 端侧人形机器人多智能体推演 demo ⬅ **沐曦基金阶段**
- [ ] **阶段 5** — 外置记忆 (.m3 三段式快照 + 三级索引) + 世界模型 demo

### 5.4 申报材料索引

- 项目主页: [README.md](../README.md)
- 架构白皮书: [architecture.md](architecture.md)
- 理论推导: [theory.md](theory.md)
- 国产 GPU 适配方案: [mxmaca_adaptation.md](mxmaca_adaptation.md)
- 完整技术报告: [paper/technical_report.md](../paper/technical_report.md)
- 实验数据汇总: [paper/data_summary.md](../paper/data_summary.md)
- 论文 LaTeX 源: [paper/main.tex](../paper/main.tex)

---

## 6. 联系与引用

### 6.1 引用

```bibtex
@misc{threechainmamba2026,
  title={ThreeChainMamba2: Three-Chain DNA-Mamba2 for Long-Range Spatiotemporal Extrapolation},
  author={lulululudj},
  year={2026},
  url={https://github.com/lulululudj/trihelix-mamba}
}
```

论文 (arXiv 预印本) 即将上线, LaTeX 源见 [paper/](../paper/)。

### 6.2 许可证

MIT — 见 [LICENSE](../LICENSE)。

### 6.3 复现指南

详见 [REPRODUCE.md](../REPRODUCE.md) 与项目根 [README.md](../README.md)。所有实验支持随机种子复现:

- 早期基准: 5 seed `[42, 123, 456, 789, 1024]`
- 大规模实验: `[0, 1, 2]`

非退化门 (模型必须通过才能进基准测试):

- `zero_ratio < 0.90` (不能全猜 0)
- `changed_acc > 0.30` (比随机 1/16 = 0.0625 强)
- OOD 非零预测 > 目标的 5%

---

> 本工作面向沐曦青年开源专项基金申请开源。核心代码 (`models/`、`train.py`、`eval_ood.py`)、实验指标 (`results_stage2/`、`results_cloud/`)、论文源码与图表 (`paper/`) 均已开源, 便于评审复现与国产 GPU 适配验证。
