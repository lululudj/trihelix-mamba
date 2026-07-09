#!/bin/bash
# setup/setup_env.sh — WSL2 内一键环境搭建 (Windows + RTX 4060 cc8.9)
#
# 用法 (在 WSL Ubuntu 里):
#   cd /mnt/e/three_chain_v3
#   bash setup/setup_env.sh
#
# 完成后:
#   bash setup/verify_env.sh        # 环境自测
#   bash setup/download_weights.sh  # 下载预训练权重
#   bash setup/verify_ood.sh        # 跑 OOD 长程评估验证
#
# 估计耗时: ~15 分钟 (含 mamba_ssm 编译 ~5 分钟)

set -e

# 进入项目根 (脚本在 setup/ 子目录)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  ThreeChainMamba2 环境搭建"
echo "  项目根: $PROJECT_ROOT"
echo "=========================================="

# ============ 1. 检查 WSL2 + NVIDIA 驱动透传 ============
echo
echo "=== [1/6] 检查 WSL2 + GPU 透传 ==="
if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ nvidia-smi 不可用"
    echo "   请先在 Windows 端安装 NVIDIA 驱动 (WSL2 会自动透传 GPU):"
    echo "   https://www.nvidia.com/Download/index.aspx"
    echo "   装完重启 Windows,再进 WSL 跑本脚本"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "✅ GPU 透传正常"

# ============ 2. 装 miniconda3 (如果没有) ============
echo
echo "=== [2/6] 检查 miniconda3 ==="
CONDA_BASE="$HOME/miniconda3"
if [ ! -d "$CONDA_BASE" ]; then
    echo "miniconda3 未安装,开始下载安装..."
    cd /tmp
    # 清华源镜像 (比官方快很多)
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_BASE"
    rm miniconda.sh
    cd "$PROJECT_ROOT"
    echo "✅ miniconda3 已装到 $CONDA_BASE"
else
    echo "✅ miniconda3 已存在"
fi

# 激活 conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 配置清华源 (复用 _wsl_setup/condarc.txt + pip.conf)
echo "配置清华源..."
mkdir -p "$HOME/.conda" "$HOME/.pip"
cp "$PROJECT_ROOT/_wsl_setup/condarc.txt" "$HOME/.condarc" 2>/dev/null || true
cp "$PROJECT_ROOT/_wsl_setup/pip.conf" "$HOME/.pip/pip.conf" 2>/dev/null || true
# pip 配置也放到 conda env 里
mkdir -p "$HOME/.config/pip"
cp "$PROJECT_ROOT/_wsl_setup/pip.conf" "$HOME/.config/pip/pip.conf" 2>/dev/null || true

# ============ 3. 创建 conda env mamba ============
echo
echo "=== [3/6] 创建 conda env mamba (Python 3.11) ==="
if conda env list | grep -q "^mamba "; then
    echo "✅ conda env mamba 已存在"
else
    echo "创建 conda env mamba..."
    conda create -n mamba python=3.11 -y -q
    echo "✅ conda env mamba 已创建"
fi
conda activate mamba

# 装 gcc-12 (mamba_ssm 编译需要,Ubuntu 22.04 默认 gcc-11)
if ! command -v gcc-12 &>/dev/null; then
    echo "装 gcc-12 (mamba_ssm 编译需要)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq gcc-12 g++-12
fi

# ============ 4. 装 torch (CUDA 12.1 build,RTX 4060 兼容) ============
echo
echo "=== [4/6] 安装 torch + 基础依赖 (CUDA 12.1 build) ==="
# RTX 4060 = sm_89 (Ada Lovelace), CUDA 12.1/12.4/12.6 都支持
# 用 PyTorch 官方 cu121 index (最稳定, mamba_ssm 2.2.x 兼容性好)
pip install -q --index-url https://download.pytorch.org/whl/cu121 \
    torch numpy pyyaml matplotlib python-dotenv

echo "验证 torch..."
python -c "
import torch
assert torch.cuda.is_available(), '❌ torch.cuda 不可用'
print(f'✅ torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}')
"

# ============ 5. 装 triton + mamba_ssm + causal_conv1d ============
echo
echo "=== [5/6] 安装 triton + mamba_ssm + causal_conv1d ==="
echo "装 triton (mamba_ssm CUDA kernel 依赖)..."
pip install -q triton

# 复用 _wsl_setup/install_mamba.sh (已针对 RTX 4060 写好: TORCH_CUDA_ARCH_LIST=8.9, CUDA 12.x, gcc-12)
# install_mamba.sh 会设 CUDA_HOME/PATH/CC/CXX 并 --no-build-isolation 编译
echo "调用 _wsl_setup/install_mamba.sh 编译安装 causal_conv1d + mamba_ssm (约 5 分钟)..."
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1
bash "$PROJECT_ROOT/_wsl_setup/install_mamba.sh"

# ============ 6. 自测 ============
echo
echo "=== [6/6] 自测 ==="
echo "--- torch + GPU ---"
python "$PROJECT_ROOT/_wsl_setup/test_torch.py"

echo
echo "--- mamba_ssm.Mamba2 ---"
python "$PROJECT_ROOT/_wsl_setup/test_mamba.py"

echo
echo "=========================================="
echo "✅ 环境搭建完成!"
echo "=========================================="
echo
echo "下一步:"
echo "  1. bash setup/download_weights.sh   # 下载预训练权重 best.pt"
echo "  2. bash setup/verify_env.sh         # 完整环境自测 (含项目 import)"
echo "  3. bash setup/verify_ood.sh          # 跑 OOD 长程评估 (核心验证)"
echo
echo "之后 (可选):"
echo "  bash setup/gen_data.sh              # 生成训练数据"
echo "  bash setup/train_main.sh            # 本机从零训练 3.26M 模型"
