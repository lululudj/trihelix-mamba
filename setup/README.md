# ThreeChainMamba2 本机部署指南

> 让任何人在 Windows 11 + RTX 4060 (cc8.9 Ada) 本机上,通过 WSL2 一键跑通 ThreeChainMamba2 环境,完成 OOD 长程外推评估验证。

---

## 前置条件

| 项 | 要求 | 检查命令 |
|---|---|---|
| 操作系统 | Windows 11 (或 Windows 10 21H2+) | `winver` |
| WSL2 | Ubuntu 22.04 | `wsl -l -v` (VERSION 应为 2) |
| NVIDIA 驱动 | 最新版 (WSL2 GPU 透传) | 在 WSL 里跑 `nvidia-smi` |
| GPU | RTX 4060 (8GB, cc8.9 Ada Lovelace) | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| 磁盘空间 | ≥ 5 GB (conda + 依赖 + 权重) | — |

### 如果 WSL2 未装
```powershell
# 在 Windows PowerShell (管理员) 里
wsl --install -d Ubuntu-22.04
# 重启后设置 Ubuntu 用户名密码
```

### 如果 NVIDIA 驱动未装
到 [NVIDIA 官网](https://www.nvidia.com/Download/index.aspx) 下载并安装最新 Game Ready 或 Studio 驱动 (WSL2 会自动透传 GPU,无需在 WSL 内单独装驱动)。

---

## 3 步走部署

### Step 1: 装环境 (~15 分钟)

在 **WSL Ubuntu** 终端里:
```bash
# 进入项目 (假设项目在 Windows 的 E:\three_chain_v3)
cd /mnt/e/three_chain_v3

# 一键装环境 (conda env mamba + torch + triton + mamba_ssm)
bash setup/setup_env.sh
```

**这个脚本会做什么:**
1. 检查 WSL2 + NVIDIA GPU 透传 (`nvidia-smi`)
2. 装 miniconda3 (如果没有),配清华源
3. 创建 conda env `mamba` (Python 3.11)
4. 装 gcc-12 (mamba_ssm 编译需要)
5. 装 torch (CUDA 12.1 build) + numpy + pyyaml + matplotlib + python-dotenv
6. 装 triton
7. 调用 `_wsl_setup/install_mamba.sh` 编译 mamba_ssm + causal_conv1d (约 5 分钟)
8. 自测 torch + mamba_ssm

**完成后环境自测:**
```bash
bash setup/verify_env.sh
# 期望输出:
# ✅ torch=2.x cuda=12.1 gpu=NVIDIA GeForce RTX 4060
# ✅ triton=2.x
# ✅ mamba_ssm=2.2.x (Mamba2 实例化 OK)
# ✅ causal_conv1d
# ✅ ThreeChainMamba2 import OK (params=3.26M)
# ✅ 环境自测全部通过
```

---

### Step 2: 下载预训练权重 (~30 秒)

```bash
bash setup/download_weights.sh
# 期望输出:
# ✅ best.pt 下载完成 (39000000 bytes, 37 MB)
```

`best.pt` = 3.26M GridWorld 主力模型,GitHub Release 提供 (39 MB)。

如果 GitHub 不通 (国内网络),脚本会提示手动从 [GitLink Release](https://www.gitlink.org.cn/lulululudj/ThreeChainMamba/releases) 下载。

---

### Step 3: 跑 OOD 长程评估验证 (~10 秒)

```bash
bash setup/verify_ood.sh
```

**期望输出:**
```
=== OOD 长程外推评估验证 ===
...
=== OOD 结果 (three_chain_mamba2) ===
acc_final (t=150): 0.5xxx
changed_acc (t=150): 0.5952
OOD 衰减 (t=100→150): -0.0042 (-0.4%)
  changed_acc@t=50: 0.6xxx
  changed_acc@t=100: 0.6xxx
  changed_acc@t=120: 0.5xxx
  changed_acc@t=150: 0.5xxx

=== 验证指标 ===
模型: three_chain_mamba2
样本数: 72, max_T: 150
changed_acc (t=150): 0.5952
ood_decay_pct (t=100→150): -0.42%

==================================================
✅ 本机部署验证通过!
   changed_acc = 0.5952 (期望 ≈ 0.5952)
   ood_decay   = -0.42% (期望 ≈ 0%)
   样本数 = 72
==================================================
```

**看到 `✅ 本机部署验证通过` 即说明环境 + 权重 + 数据全部就绪,可以开始训练/评估了。**

---

## 常见问题

### Q1: `nvidia-smi` 在 WSL 里报 "command not found"
**原因**: Windows 端未装 NVIDIA 驱动,或驱动版本太旧。
**解决**:
1. 到 [NVIDIA 官网](https://www.nvidia.com/Download/index.aspx) 下载最新驱动
2. 安装后重启 Windows
3. 再进 WSL 跑 `nvidia-smi` 应能看到 GPU

### Q2: `mamba_ssm` 编译失败 (nvcc 报错)
**原因**: CUDA 版本与 torch 不匹配,或 gcc 版本不对。
**解决**:
```bash
# 在 WSL 里检查
nvcc --version    # 期望 12.x
python -c "import torch; print(torch.version.cuda)"   # 应与 nvcc 一致
gcc-12 --version  # 必须有
```
如果 torch.cuda 与 nvcc 版本不一致,重装 torch:
```bash
pip uninstall torch
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

### Q3: `best.pt` 下载失败 (GitHub 不通)
**解决**: 用 GitLink 镜像
1. 访问 https://www.gitlink.org.cn/lulululudj/ThreeChainMamba/releases
2. 下载 `best.pt`
3. 放到 `E:\three_chain_v3\best.pt`
4. 再跑 `bash setup/verify_ood.sh`

### Q4: 训练时 OOM (显存不足)
**原因**: RTX 4060 只有 8GB 显存,大模型可能 OOM。
**解决**:
```bash
# 减小 batch_size
python train.py --config configs/matched_mamba2.yaml --batch_size 2 ...
# 或开 gradient checkpointing (1B 模型必须)
# configs/matched_mamba2_1000m.yaml 已含 use_checkpoint: true
```

### Q5: `conda activate mamba` 报错
**原因**: conda 未初始化。
**解决**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba
```
或一劳永逸:
```bash
conda init bash
# 重开终端后 conda activate mamba 就能用
```

### Q6: WSL2 里访问 E 盘项目路径
WSL2 把 Windows 盘符挂在 `/mnt/` 下:
- `E:\three_chain_v3` → `/mnt/e/three_chain_v3`
- `C:\Users\xxx` → `/mnt/c/Users/xxx`

---

## 下一步

环境验证通过后,可选进阶:

### 完整训练复现 (P1)
```bash
bash setup/gen_data.sh        # 生成 GridWorld 训练数据 (~30 秒)
bash setup/train_main.sh      # 从零训练 3.26M 主力模型 (~10-15 分钟)
bash setup/verify_gate.sh     # 非退化门验证 (论文基准准入门槛)
```

### 5-seed 基准 + 消融实验 (P2)
```bash
bash setup/run_5seeds.sh      # 5 seed 完整基准 (~1-1.5 小时)
bash setup/run_ablations.sh   # BPv1/BPv2/Transformer 消融实验
```

### 详细文档
- [项目主 README](../README.md) — 完整项目说明 + 实验结果
- [复现指南 REPRODUCE.md](../REPRODUCE.md) — 5-seed 基准 + 消融实验复现步骤
- [部署计划文档](../.trae/documents/本机部署方案-Windows+RTX4060.md) — 本部署方案的设计文档

---

## 文件清单

```
setup/
├── README.md              # 本文档 (你正在看的)
├── setup_env.sh           # 一键环境搭建 (P0)
├── verify_env.sh          # 环境自测 (P0)
├── download_weights.sh    # 下载 best.pt (P0)
├── verify_ood.sh          # OOD 长程评估验证 (P0, 核心交付物)
├── gen_data.sh            # 生成训练数据 (P1)
├── train_main.sh          # 训练 3.26M 主力模型 (P1)
├── verify_gate.sh         # 非退化门验证 (P1)
├── run_5seeds.sh          # 5-seed 完整基准 (P2)
├── run_ablations.sh       # 消融实验 (P2)
└── environment.yml        # conda 环境固化 (P2)
```
