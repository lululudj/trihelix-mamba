# 三链 DNA-Mamba3 部署指南

> 本文档介绍 ThreeChainMamba3 项目的环境搭建、数据生成、本地训练、C500 国产 GPU 部署、断点续跑及常见问题排查。
> 关联代码: [train.py](../train.py) · [battle32_c500_gpu.py](../battle32_c500_gpu.py) · [c500_plink.py](../c500_plink.py)

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装步骤](#2-安装步骤)
3. [配置说明](#3-配置说明)
4. [数据生成](#4-数据生成)
5. [本地训练运行方法](#5-本地训练运行方法)
6. [C500 国产 GPU 部署方法](#6-c500-国产-gpu-部署方法)
7. [断点续跑机制说明](#7-断点续跑机制说明)
8. [结果下载和分析方法](#8-结果下载和分析方法)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 环境要求

### 1.1 基本要求

| 项目 | 要求 |
|---|---|
| Python | >= 3.10 |
| 操作系统 | Linux (推荐) / Windows WSL2 / 国产 GPU 环境 |
| GPU | NVIDIA CUDA GPU 或 沐曦 C500 (MXMACA) |
| PyTorch | >= 2.0.0 |
| mamba_ssm | >= 2.2.0 (Mamba2 baseline) / >= 2.3.0 (Mamba3, 含官方 Triton 内核) |

### 1.2 GPU 选项

| 平台 | 显存要求 | 说明 |
|---|---|---|
| NVIDIA GPU (本地/WSL2) | >= 4GB (d_model=64) | 官方 Mamba3 Triton 内核, 最快 |
| 沐曦 C500 (MXMACA) | >= 8GB | mamba3_ref 纯 Python 版, 稳定 |
| CPU only | 无限制 | mamba3_ref 纯 Python, 仅用于调试 |

### 1.3 依赖列表

```
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
mamba_ssm>=2.2.0
causal_conv1d>=1.2.0
triton>=2.0.0
python-dotenv>=1.0.0
paramiko>=3.0.0
scipy>=1.10.0
```

---

## 2. 安装步骤

### 2.1 基础环境安装

```bash
# 克隆项目
git clone <仓库地址>
cd three_chain_v3

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2.2 mamba_ssm 编译安装 (关键)

mamba_ssm 需要从源码编译，注意以下事项：

```bash
# 安装 triton (必须先于 mamba_ssm)
pip install triton>=2.0.0

# 从源码编译 mamba_ssm
git clone https://github.com/state-spaces/mamba.git
cd mamba
# ★ --no-build-isolation: 复用已安装的 torch 版本编译, 否则可能拉取不兼容的 torch
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --no-deps -e .

# 同样方式安装 causal_conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install --no-build-isolation --no-deps -e .
```

**编译注意事项**:
- `--no-build-isolation` 是必须的：确保使用当前环境已安装的 torch 版本，否则 pip 可能拉取不兼容的 torch
- `--no-deps` 避免重复安装依赖
- `-e` 开发模式安装，方便后续更新
- 编译需要 CUDA toolkit (nvcc)，确认 `nvcc --version` 可用

### 2.3 Mamba3 开发模式 (PYTHONPATH)

如果系统已安装 mamba_ssm 2.2.4 (Mamba2 baseline) 但需要 Mamba3，可通过 PYTHONPATH 加载 2.3.x：

```bash
# 下载 mamba_ssm 2.3.x 源码到本地
# 通过 PYTHONPATH 指定优先加载路径
export PYTHONPATH=/path/to/mamba_official:$PYTHONPATH

# 验证
python -c "import mamba_ssm; print(mamba_ssm.__version__)"  # 应输出 2.3.x
python -c "from mamba_ssm import Mamba3; print('Mamba3 OK')"
```

### 2.4 验证安装

```bash
# 快速验证 mamba_ssm + Mamba3 + PairwiseBasePairMamba3
python c500_smoke.py
# 期望输出:
#   torch: <version>, cuda: True
#   mamba_ssm: <version>
#   Mamba3 OK
#   Mamba3 GPU forward+backward OK
#   PairwiseBasePairMamba3 OK
#   ThreeChainMamba3+BP forward OK
```

---

## 3. 配置说明

### 3.1 .env 文件配置 (C500 SSH 连接)

在项目根目录创建 `.env` 文件 (不会提交到 git)：

```bash
# C500 SSH 连接配置
C500_SSH_HOST=your.host.ip
C500_SSH_PORT=32222
C500_SSH_USER=your_username
C500_SSH_PASS=your_password
C500_SSH_HOSTKEY=SHA256:xxxxx
```

也可直接设置系统环境变量：

```bash
# Linux/macOS
export C500_SSH_HOST=your.host.ip
export C500_SSH_PORT=32222
export C500_SSH_USER=your_username
export C500_SSH_PASS=your_password
export C500_SSH_HOSTKEY=SHA256:xxxxx

# Windows PowerShell
$env:C500_SSH_HOST="your.host.ip"
$env:C500_SSH_PORT="32222"
$env:C500_SSH_USER="your_username"
$env:C500_SSH_PASS="your_password"
$env:C500_SSH_HOSTKEY="SHA256:xxxxx"
```

### 3.2 环境变量说明

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MAMBA3_FORCE_REF` | `0` | `1`=强制使用 mamba3_ref 纯 Python 版 (C500 用) |
| `MAMBA3_FORCE_GPU_SCAN` | `0` | `1`=mamba3_ref 的 scan 在 GPU 上跑 (C500 用) |
| `MAMBA3_USE_TRITON_SCAN` | `0` | `1`=用 selective_scan_fn Triton 内核加速 (仅 SISO) |
| `MACA_PATH` | (无) | MXMACA 安装路径, 存在时自动启用 GPU scan |
| `PYTORCH_MACA_ALLOC_CONF` | (无) | MXMACA 内存分配配置 |
| `C500_SSH_*` | (无) | C500 SSH 连接参数 (见 3.1) |

### 3.3 YAML 配置文件

主配置文件 `configs/default.yaml`:

```yaml
# 任务设定
task:
  cell_types: 16          # C: 空格(0) + 智能体ID(1-12) + 物品ID(13-15)
  action_dim: 5           # A: 上/下/左/右/停留

# 模型默认
model:
  d_model: 256            # 每条链的隐藏维度
  n_layers: 2             # 每条链的 Mamba 层数
  bind_heads: 4           # CrossAttn 头数
  fusion_hidden: 512

# 训练
train:
  batch_size: 16
  max_steps: 50000
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  loss_aux_weight: 0.3
  grad_clip: 1.0
  seed: 0
```

battle32 实验使用独立配置 (在 `battle32_c500_gpu.py` 中硬编码):

```python
D_MODEL = 64
BATCH_SIZE = 1
MAX_STEPS = 100
LR = 1e-3
AUX_WEIGHT = 0.3
SEEDS = [42, 123, 456]
N_TRAIN, N_VAL, N_OOD = 60, 16, 16
```

---

## 4. 数据生成

### 4.1 GridWorld 数据生成器

使用 `data/gen_grid_world.py` 生成 GridWorld 场景数据：

```bash
# 生成完整数据集 (81 子配置 × 125 场景 = 10125 样本)
python data/gen_grid_world.py --out_dir ./data_cache --seed 0

# 生成指定配置的数据
python data/gen_grid_world.py \
    --out_dir ./data_custom \
    --N 8 --K 8 --T 100 --p_transfer 0.15 \
    --scenario_type goal_directed \
    --num_per_config 100 \
    --seed 0
```

### 4.2 32 场景数据生成

battle32 实验的数据在 `battle32_c500_gpu.py` 运行时**自动生成**，无需手动执行：

```python
# battle32_c500_gpu.py 中的自动数据生成
def prepare_scenario_data(name, N, K, T_train, T_ood, p_transfer, scenario_type):
    scen_dir = DATA_ROOT / name
    for split, n, T, seed_base in [
        ("train", N_TRAIN, T_train, hash(name) % 10000),
        ("val",   N_VAL,   T_train, 100000 + hash(name) % 10000),
        ("ood",   N_OOD,   T_ood,   200000 + hash(name) % 10000),
    ]:
        out = scen_dir / split
        if out.exists() and any(out.glob("scen_*.npz")):
            continue  # 已存在则跳过
        gen_scen_files(out, n, N, K, T, p_transfer, scenario_type, seed_base)
```

数据存储路径: `data_battle32/<场景名>/{train,val,ood}/scen_XXXXX.npz`

### 4.3 数据格式

每个 `.npz` 文件包含:

| 键 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `S_0` | (N, N) | int64 | 初始网格状态 |
| `actions` | (K, T) | int64 | K 个 agent 的 T 步动作序列 |
| `S_t` | (T+1, N, N) | int64 | 完整轨迹 (含初始状态) |
| `N` | 标量 | int | 网格尺寸 |
| `K` | 标量 | int | agent 数量 |
| `T` | 标量 | int | 序列长度 |
| `p_transfer` | 标量 | float | 冲突时物品转移概率 |
| `scenario_type` | 标量 | str | 场景类型 |

### 4.4 数据集划分

```bash
# 将 data_cache 划分为 train/val/test
python -c "import sys; sys.path.insert(0,'.'); \
from data.dataset import split_dataset; \
split_dataset('./data_cache', './data_split')"
```

---

## 5. 本地训练运行方法

### 5.1 标准 train.py 训练

```bash
# 快速验证 (100 步)
python train.py --model three_chain_mamba3 --seed 0 --max_steps 100

# 正式训练 (使用默认配置)
python train.py --model three_chain_mamba3 --seed 0

# 自定义配置
python train.py \
    --model three_chain_mamba3 \
    --seed 42 \
    --config configs/default.yaml \
    --data_root ./data_split \
    --batch_size 16 \
    --max_steps 50000 \
    --eval_every 1000 \
    --log_every 50 \
    --save_every 5000 \
    --out_dir results/run_mamba3_seed42
```

### 5.2 可用模型列表

```bash
# 支持的模型 (--model 参数)
three_chain              # 原始三链 (Mamba1)
three_chain_mamba2       # 三链 Mamba2 (SSD)
three_chain_mamba2_bp    # 三链 Mamba2 + BP v1
three_chain_mamba2_bpv2  # 三链 Mamba2 + BP v2
three_chain_mamba3       # 三链 Mamba3 (梯形离散化 + dt-RoPE)
single_chain             # 单链 baseline
single_chain_mamba3      # Mamba3 单链 baseline
transformer              # Transformer baseline
gnn                      # GNN baseline
```

### 5.3 battle32 本地运行

```bash
# 本地 GPU 运行 32 场景实验 (需 CUDA GPU)
python battle32_gpu.py

# C500 加固版 (含 cuda.synchronize 防护, BATCH_SIZE=1)
python battle32_c500_gpu.py
```

### 5.4 断点续跑

train.py 支持 `--resume` 从 checkpoint 恢复:

```bash
python train.py \
    --model three_chain_mamba3 \
    --seed 0 \
    --resume results/run_mamba3_seed0/best.pt
```

### 5.5 输出文件

训练完成后在 `out_dir` 下生成:

```
results/run_mamba3_seed0/
├── log.jsonl          # 训练日志 (每 log_every 步一条)
├── best.pt            # 最佳 val 模型 checkpoint
├── final.pt           # 最终模型 checkpoint
└── summary.json       # 训练摘要
```

---

## 6. C500 国产 GPU 部署方法

### 6.1 部署流程概览

通过 `c500_plink.py` 工具完成 C500 远程部署，全部操作通过 SSH 自动执行：

```
本地 ──upload──> C500 ──check──> 验证环境 ──launch──> 启动实验
                                                   └──progress──> 监控进度
                                                   └──download──> 下载结果
```

### 6.2 上传代码

```bash
# 上传所有必要文件到 C500
python c500_plink.py upload
# 上传的文件包括:
#   battle32_c500_gpu.py, stats_analysis.py
#   models/three_chain_mamba3.py, models/__init__.py, models/baselines.py
#   models/common.py, models/three_chain_mamba2.py, models/three_chain_mamba2_bp.py
#   models/three_chain_mamba2_bpv2.py, models/three_chain.py, models/mamba3_ref.py
#   data/gen_grid_world.py, data/dataset.py, data/__init__.py
#   c500_smoke.py, c500_run_battle32_v2.sh
```

文件通过 base64 编码 + stdin 管道传输，远程路径为 `/root/three_chain_v3/`。

### 6.3 检查环境

```bash
# 验证 C500 上的环境 (torch + mamba_ssm + Mamba3 + BP)
python c500_plink.py check
# 期望输出: "Environment check: PASS"
```

检查脚本 `c500_smoke.py` 会验证:
- PyTorch + CUDA 可用性
- mamba_ssm 版本
- Mamba3 forward + backward
- PairwiseBasePairMamba3 forward + backward
- ThreeChainMamba3+BP 完整模型

### 6.4 启动实验

```bash
# 启动 32 场景实验 (后台 nohup 运行, 断点续跑)
python c500_plink.py launch
```

启动脚本 `c500_run_battle32_v2.sh` 会:
1. 设置 MXMACA 环境变量
2. 清理残留进程
3. 以 `nohup` 后台运行 `battle32_c500_gpu.py`
4. 崩溃自动重试 (最多 10 次)
5. 完成后自动运行 `stats_analysis.py`

### 6.5 监控进度

```bash
# 检查实验进度 (查看最近 30 行日志 + 已完成数量)
python c500_plink.py progress
# 输出示例: "Completed: 85/384"
```

### 6.6 一键全流程

```bash
# 上传 + 检查 + 启动 (一步到位)
python c500_plink.py all
```

### 6.7 C500 远程环境变量

`c500_run_battle32_v2.sh` 在 C500 上设置的环境变量:

```bash
source /opt/maca/env.sh
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib
export MAMBA3_FORCE_REF=1           # 强制 mamba3_ref 纯Python版
export MAMBA3_FORCE_GPU_SCAN=1      # GPU 上跑 scan (C500 无 TDR 限制)
export MAMBA3_USE_TRITON_SCAN=0     # 禁用 Triton selective_scan
export PYTORCH_MACA_ALLOC_CONF=max_split_size_mb:128
```

---

## 7. 断点续跑机制说明

### 7.1 battle32 断点续跑

`battle32_c500_gpu.py` 内置断点续跑，基于 `battle32_results.json` 持久化：

```python
# 启动时加载已有结果
if RESULTS_PATH.exists():
    all_results = json.load(open(RESULTS_PATH))
    log(f"已有结果: {len(all_results)} 条, 断点续跑")

# 跳过已完成的实验
key = f"{scen_name}|seed{seed}|{display_name}"
if key in all_results and "error" not in all_results[key]:
    log(f"[skip] {key}")
    continue

# 每完成一个实验立即保存
all_results[key] = result
json.dump(all_results, open(RESULTS_PATH, "w"))
```

### 7.2 Shell 层面自动重试

`c500_run_battle32_v2.sh` 的重试机制:

```bash
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    setsid python3 -u battle32_c500_gpu.py >> $LOG 2>&1
    RC=$?

    # 检查是否完成
    if grep -q "C500 EXPERIMENT COMPLETE" $LOG; then
        break
    fi

    # 未完成, 重试
    RETRY=$((RETRY+1))
    sleep 15
    pkill -9 -f battle32_c500_gpu.py
    sleep 5
    mx-smi -r 2>/dev/null  # GPU reset
    sleep 3
done
```

### 7.3 train.py 断点续跑

```bash
# 从 best.pt 恢复 (含 optimizer 状态 + step + best_val)
python train.py --model three_chain_mamba3 --seed 0 \
    --resume results/run_mamba3_seed0/best.pt
```

恢复时自动读取:
- `step`: 从断点步数继续
- `best_val`: 保留最佳验证分数
- `model.state_dict()`: 模型权重
- `optimizer.state_dict()`: 优化器状态 (动量等)

---

## 8. 结果下载和分析方法

### 8.1 下载 C500 结果

```bash
# 下载结果文件到本地
python c500_plink.py download
```

下载的文件:

| 远程文件 | 本地文件 | 说明 |
|---|---|---|
| `battle32_results.json` | `battle32_c500_results.json` | 384 组实验结果 |
| `battle32_stats.json` | `battle32_c500_stats.json` | 统计检验结果 |
| `battle32_c500_log.txt` | `battle32_c500_log.txt` | 完整训练日志 |

### 8.2 结果文件格式

`battle32_results.json` 中每条记录:

```json
{
  "rand_n6_k4|seed42|三链Mamba3+BP": {
    "model": "三链Mamba3+BP",
    "seed": 42,
    "params": 177000,
    "steps": 100,
    "train_time_s": 45.2,
    "val_acc": 92.15,
    "val_ch_acc": 68.50,
    "ood_acc": 88.30,
    "ood_ch_acc": 65.20,
    "ood_decay_pct": -4.82,
    "bp_mod_strength": {
      "bp1_delta": 0.0312,
      "bp1_gate": 0.0285,
      "bp2_reset": 0.0198,
      "bp3_gamma": 0.0421,
      "bp3_beta": 0.0356
    }
  }
}
```

### 8.3 统计分析

```bash
# 运行统计检验 (t-test + Cohen's d + 95% CI)
python stats_analysis.py
# 或指定结果文件
python stats_analysis.py battle32_c500_results.json
```

输出:
- `battle32_stats.json`: 统计检验结果 (逐场景 + 全场景汇总)
- 控制台汇总表: treatment (三链Mamba3+BP) vs 3 个对照组的 t/p/d/显著性

### 8.4 BP 调制强度分析

```bash
# 检查 BP 是否在学习 (modulation_strength 非零)
python _bp_v21_analysis.py
```

该脚本检查下载结果中 BP 模型的 `bp_mod_strength` 字段，验证 `bp1_gate` 和 `bp2_reset` 是否从零初始化增长到非零值 (v2.1 修复有效性的证据)。

### 8.5 可视化

```bash
# 生成对比图表
python plot_bp_compare.py     # BP vs 无 BP 对比图
python plot_comparison.py     # 多模型对比图
python generate_figures.py    # 论文用图表
```

---

## 9. 常见问题排查

### 9.1 `__pycache__` 缓存问题

**症状**: 修改了代码但行为不变，或导入旧版本模块。

**原因**: Python 缓存了 `.pyc` 文件，修改源码后未重新编译。

**解决**:

```bash
# 清除所有 __pycache__ 目录
find . -type d -name __pycache__ -exec rm -rf {} +

# Windows PowerShell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

**预防**: 在 C500 上传代码前，确保本地代码已保存最新版本。`c500_plink.py` 上传时会覆盖远程文件，但 `__pycache__` 不会被清除。

### 9.2 Triton 后端问题

**症状**: `ImportError: cannot import name 'Mamba3' from 'mamba_ssm'` 或 Triton 编译失败。

**原因**: 系统 mamba_ssm 版本过低 (2.2.x 无 Mamba3)，或 Triton 与 CUDA 版本不兼容。

**解决**:

```bash
# 方案1: 使用 PYTHONPATH 加载 mamba_ssm 2.3.x
export PYTHONPATH=/path/to/mamba_official:$PYTHONPATH

# 方案2: 强制使用 mamba3_ref 纯 Python 版 (无需 Triton)
export MAMBA3_FORCE_REF=1

# 方案3: C500 上禁用 Triton selective_scan, 用纯 Python GPU scan
export MAMBA3_USE_TRITON_SCAN=0
export MAMBA3_FORCE_GPU_SCAN=1
```

**验证**:

```bash
python -c "from mamba_ssm import Mamba3; print('Mamba3 OK')"
# 或
python c500_smoke.py
```

### 9.3 OOM (显存不足) 处理

**症状**: `RuntimeError: CUDA out of memory` 或 C500 上的 MACA OOM。

**解决**:

```bash
# 方案1: 减小 batch_size (battle32 已强制 BATCH_SIZE=1)
# 在 train.py 中:
python train.py --model three_chain_mamba3 --batch_size 1

# 方案2: 启用梯度检查点 (大模型)
# 在 config YAML 中:
#   model:
#     use_checkpoint: true

# 方案3: 减小 d_model
#   model:
#     d_model: 64   # 从 256 降到 64

# 方案4: MXMACA 内存分配优化
export PYTORCH_MACA_ALLOC_CONF=max_split_size_mb:128
```

**battle32 中的 OOM 自动恢复**:

```python
if "out of memory" in emsg:
    log(f"[OOM] step {step}, skip batch")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    continue  # 跳过当前 batch, 继续训练
```

### 9.4 C500 rq_qos_wait 死锁

**症状**: C500 上训练卡住不动，日志无输出。

**原因**: MXMACA GPU 驱动的 kernel 队列堆积导致死锁。

**解决**: `battle32_c500_gpu.py` 已内置防护:
- 每步训练后 `torch.cuda.synchronize()` 防止队列堆积
- 模型间切换时 `synchronize() + sleep(0.3)`
- Shell 层面崩溃自动重试 (最多 10 次) + `mx-smi -r` GPU reset

**手动恢复**:

```bash
# 通过 SSH 手动重置
python c500_plink.py check  # 重新连接检查
# 或直接 SSH 上去:
pkill -9 -f battle32_c500_gpu.py
mx-smi -r
sleep 5
python c500_plink.py launch  # 重新启动 (断点续跑)
```

### 9.5 SSH 连接失败

**症状**: `c500_plink.py` 报错 "缺少必要的 SSH 连接环境变量"。

**解决**:

```bash
# 检查 .env 文件是否存在且格式正确
cat .env
# 应包含:
#   C500_SSH_HOST=...
#   C500_SSH_USER=...
#   C500_SSH_PASS=...
#   C500_SSH_HOSTKEY=SHA256:...

# 或直接设置环境变量
export C500_SSH_HOST=your.host.ip
export C500_SSH_USER=your_username
export C500_SSH_PASS=your_password
export C500_SSH_HOSTKEY=SHA256:xxxxx
```

**注意**: `python-dotenv` 是可选依赖，未安装时需直接设置系统环境变量。

### 9.6 数据生成失败

**症状**: `FileNotFoundError: No scen_*.npz under ...`

**解决**:

```bash
# 检查数据目录是否存在且有文件
ls data_battle32/rand_n6_k4/train/

# 手动重新生成 (删除后重跑)
rm -rf data_battle32/
python battle32_c500_gpu.py  # 会自动重新生成
```

### 9.7 模型参数不匹配

**症状**: `TypeError: __init__() got an unexpected keyword argument 'max_T'`

**原因**: Mamba3 版本的模型不接受 `max_T` 参数 (Mamba3 不使用固定长度 time_embed)。

**解决**: `battle32_c500_gpu.py` 已做容错处理:

```python
try:
    model = cls(**kwargs)  # 含 max_T
except TypeError:
    kwargs.pop("max_T")
    model = cls(**kwargs)  # 不含 max_T
```

### 9.8 mamba_ssm 编译失败

**症状**: `pip install mamba_ssm` 时 CUDA 编译报错。

**排查清单**:

```bash
# 1. 检查 nvcc 版本
nvcc --version

# 2. 检查 torch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 3. 确保 nvcc 和 torch CUDA 版本一致
# 4. 确保使用 --no-build-isolation
pip install --no-build-isolation --no-deps -e .

# 5. C500 上使用 mamba3_ref 纯 Python 版, 无需编译 mamba_ssm
export MAMBA3_FORCE_REF=1
```
