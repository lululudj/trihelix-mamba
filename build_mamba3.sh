#!/bin/bash
# 在 venv 里从源码编译官方 mamba_ssm（含 Mamba3）
# 关键：CUDA_HOME 和 PATH 设置，nvcc 可用
set -e
source "$HOME/mamba3_venv/bin/activate"

echo "=== 0. 降级 setuptools 到 81.x（torch 兼容）==="
pip install 'setuptools<82' 2>&1 | tail -2

echo "=== 1. 设置 CUDA 环境 ==="
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -3
echo "CUDA_HOME=$CUDA_HOME"

echo "=== 2. 检查 venv mamba_ssm 当前状态 ==="
python3 -c "import mamba_ssm; print('当前:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 3. 从源码编译官方 mamba_ssm（含 Mamba3）==="
echo "    预计 10-30 分钟，会编译 Triton/CUDA 内核..."
echo "    命令: MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/state-spaces/mamba.git"
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/state-spaces/mamba.git 2>&1 | tail -50

echo "=== 4. 验证 venv 里 mamba_ssm 已升级 ==="
python3 -c "import mamba_ssm; print('升级后:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 5. 验证 Mamba3 可 import ==="
python3 -c "from mamba_ssm import Mamba3; print('✅ Mamba3 import 成功！'); m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16); print('实例化成功:', m)"

echo "=== DONE ==="
