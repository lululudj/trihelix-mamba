#!/bin/bash
# 在 venv 里从本地源码编译官方 mamba_ssm（含 Mamba3）
set -e
source "$HOME/mamba3_venv/bin/activate"

echo "=== 1. 设置 CUDA 环境 ==="
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -2

echo "=== 2. venv 里 mamba_ssm 当前状态 ==="
python3 -c "import mamba_ssm; print('before:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 3. 从本地路径编译官方 mamba_ssm（含 Mamba3）==="
echo "    预计 10-30 分钟..."
cd /mnt/e/mamba_official
ls setup.py mamba_ssm/modules/mamba3.py 2>/dev/null && echo "源码完整"
echo "---"
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall --no-deps . 2>&1 | tail -60

echo "=== 4. 验证 venv 里 mamba_ssm 已升级 ==="
python3 -c "import mamba_ssm; print('after:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 5. 验证 Mamba3 可 import + 实例化 ==="
python3 -c "
from mamba_ssm import Mamba3
print('✅ Mamba3 import 成功')
m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16, is_mimo=False)
print('✅ 实例化成功, params:', sum(p.numel() for p in m.parameters()))
"

echo "=== DONE ==="
