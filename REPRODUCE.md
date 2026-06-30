# Reproducibility Guide 可复现指南

🇬🇧 English | 🇨🇳 [中文](#中文-1)

This guide lets anyone clone the repo, run the experiments, and verify every number in the paper/report.

---

## English

### 1. Environment

**Requirements:**
- NVIDIA GPU with CUDA (8GB+ VRAM; tested on RTX 3060 Laptop 6GB/8GB)
- Python 3.10–3.12 (Python 3.14 may lack prebuilt `mamba_ssm` wheels)
- Linux/WSL2 recommended (Mamba2 needs CUDA kernels)

```bash
# Clone
git clone https://github.com/lululudj/trihelix-mamba.git
cd trihelix-mamba

# Install
pip install torch numpy pyyaml matplotlib
pip install triton mamba_ssm causal_conv1d
```

> If `mamba_ssm` install fails, see [mamba_ssm repo](https://github.com/state-spaces/mamba) for CUDA/torch version matching.

### 2. Data

**OOD test data is INCLUDED** in `data/ood_T150/` (72 samples, T=150) — you can run OOD evaluation immediately.

**Training data** needs to be generated:

```bash
# Generate GridWorld training data (N=6/8/12, K=4/8/12, T=30/60/100)
python data/gen_grid_world.py

# Generate OOD test data (if you want to regenerate, optional — already included)
python data/gen_ood.py
```

### 3. Train the Main Model

```bash
# Train ThreeChainMamba2 (3.26M params), 2000 steps, seed=42
python train.py --config configs/matched_mamba2.yaml --model three_chain_mamba2 --seed 42

# Or use the degradation diagnosis tool (trains + evaluates OOD in one go)
python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 2000 --seed 42 --batch_size 4
```

### 4. Run All 5 Seeds (Full Benchmark)

```bash
for SEED in 42 123 456 789 1024; do
    python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 2000 --seed $SEED --batch_size 4
done
```

### 5. Ablation Experiments

```bash
# BPv1: in-chain base pair (should hurt OOD by ~5.6%)
python probe_mamba2_brain.py --model three_chain_mamba2_bp --max_steps 2000 --seed 42 --batch_size 4

# BPv2: cross-chain base pair (should tie with baseline)
python probe_mamba2_brain.py --model three_chain_mamba2_bpv2 --max_steps 2000 --seed 42 --batch_size 4

# Transformer-tiny: same params (should collapse to all-zeros!)
python probe_mamba2_brain.py --model transformer --config configs/matched_transformer_tiny.yaml --max_steps 2000 --seed 42 --batch_size 4
```

### 6. Verify Results

All experiment metrics are included in `results_wsl/`:

```
results_wsl/
├── benchmark_mamba2/          # Main model (5 seeds)
│   └── three_chain_mamba2_seed{42,123,456,789,1024}/
│       ├── summary.json       # Training summary
│       ├── ood_metrics.json   # OOD evaluation results
│       └── log.jsonl          # Training log (per-step)
└── benchmark_multiseed/       # Baselines (old ThreeChain, SingleChain, Transformer)
```

**Key numbers to verify:**

| Metric | Expected Value | Source |
|---|---|---|
| ThreeChainMamba2 OOD ch_acc@150 | 0.5952 ± 0.0034 | `benchmark_mamba2/*/ood_metrics.json` |
| ThreeChainMamba2 OOD decay | −0.0042 | same |
| Transformer-tiny OOD zero_ratio | 1.000 (collapse) | run experiment |
| Old ThreeChain 5-seed std | 0.0000 (bug) | `benchmark_multiseed/three_chain_seed*/ood_metrics.json` |

### 7. Regenerate Figures

```bash
python regen_figures.py
# Outputs: figures/fig1..fig7.png (cosmic dark style, ASCII-safe)
```

### 8. Non-Degeneration Gate

A model must pass before benchmarking:
- `zero_ratio < 0.90` (not all-zeros)
- `changed_acc > 0.30` (better than random 1/16)
- OOD non-zero predictions > 5% of target

---

<a id="中文-1"></a>
## 中文

### 1. 环境

**要求：**
- NVIDIA GPU + CUDA（8GB+ 显存；在 RTX 3060 6GB/8GB 测试通过）
- Python 3.10–3.12（3.14 可能没有 `mamba_ssm` 预编译包）
- 推荐 Linux/WSL2（Mamba2 需要 CUDA 内核）

```bash
git clone https://github.com/lululudj/trihelix-mamba.git
cd trihelix-mamba

pip install torch numpy pyyaml matplotlib
pip install triton mamba_ssm causal_conv1d
```

### 2. 数据

**OOD 测试数据已包含**在 `data/ood_T150/`（72 个样本，T=150）——可以直接跑 OOD 评估。

**训练数据**需要生成：

```bash
python data/gen_grid_world.py   # 生成训练数据
python data/gen_ood.py          # 重新生成 OOD 数据（可选，已包含）
```

### 3. 训练主力模型

```bash
python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 2000 --seed 42 --batch_size 4
```

### 4. 跑 5 个种子（完整基准）

```bash
for SEED in 42 123 456 789 1024; do
    python probe_mamba2_brain.py --model three_chain_mamba2 --max_steps 2000 --seed $SEED --batch_size 4
done
```

### 5. 消融实验

```bash
# BPv1：链内碱基对（OOD 应降 ~5.6%）
python probe_mamba2_brain.py --model three_chain_mamba2_bp --max_steps 2000 --seed 42 --batch_size 4

# BPv2：跨链碱基对（应与 baseline 持平）
python probe_mamba2_brain.py --model three_chain_mamba2_bpv2 --max_steps 2000 --seed 42 --batch_size 4

# Transformer-tiny：同参数（应完全退化成全猜 0！）
python probe_mamba2_brain.py --model transformer --config configs/matched_transformer_tiny.yaml --max_steps 2000 --seed 42 --batch_size 4
```

### 6. 验证结果

所有实验指标已包含在 `results_wsl/`：

```
results_wsl/
├── benchmark_mamba2/          # 主力模型（5 种子）
│   └── three_chain_mamba2_seed{42,123,456,789,1024}/
│       ├── summary.json       # 训练摘要
│       ├── ood_metrics.json   # OOD 评估结果
│       └── log.jsonl          # 训练日志（逐步）
└── benchmark_multiseed/       # 基线（旧 ThreeChain, SingleChain, Transformer）
```

**待验证的关键数字：**

| 指标 | 期望值 | 来源 |
|---|---|---|
| ThreeChainMamba2 OOD ch_acc@150 | 0.5952 ± 0.0034 | `benchmark_mamba2/*/ood_metrics.json` |
| ThreeChainMamba2 OOD 衰减 | −0.0042 | 同上 |
| Transformer-tiny OOD zero_ratio | 1.000（塌缩） | 运行实验 |
| 旧 ThreeChain 5 种子 std | 0.0000（bug 假象） | `benchmark_multiseed/three_chain_seed*/ood_metrics.json` |

### 7. 重新生成图表

```bash
python regen_figures.py
# 输出：figures/fig1..fig7.png（宇宙深色风，ASCII 安全）
```

### 8. 非退化门

模型必须通过才能进基准测试：
- `zero_ratio < 0.90`（不能全猜 0）
- `changed_acc > 0.30`（比随机 1/16 强）
- OOD 非零预测 > 目标的 5%
