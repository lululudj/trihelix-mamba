#!/bin/bash
# 建独立 venv，复用系统 torch/triton，隔离装官方 mamba_ssm（含 Mamba3）
# 不动现有 .local 里的 mamba_ssm 2.2.4
set -e
cd /mnt/e/three_chain_v3

VENV_DIR="$HOME/mamba3_venv"
echo "=== 1. 建 venv (复用系统 site-packages) ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "venv 已建在 $VENV_DIR"
else
    echo "venv 已存在，跳过创建"
fi

# 激活 venv
source "$VENV_DIR/bin/activate"

echo "=== 2. 验证 venv 能看到系统 torch/triton ==="
which python3
python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import triton; print('triton:', triton.__version__)"
python3 -c "import einops; print('einops ok')"
python3 -c "import yaml; print('pyyaml ok')"
python3 -c "import numpy; print('numpy:', numpy.__version__)"

echo "=== 3. 验证 venv 里 mamba_ssm 当前是 2.2.4 ==="
python3 -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__, '| path:', mamba_ssm.__file__)"

echo "=== 4. 升级 pip + wheel ==="
pip install --upgrade pip wheel setuptools 2>&1 | tail -3

echo "=== DONE step1: venv ready ==="
