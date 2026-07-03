#!/bin/bash
# 加 --no-build-isolation 重新编译（用 venv 里已有的 torch CUDA）
set -e
source "$HOME/mamba3_venv/bin/activate"

echo "=== 1. 装 build 工具（ninja/packaging/wheel）==="
pip install --no-deps ninja packaging wheel 2>&1 | tail -3
python3 -c "import ninja; print('ninja:', ninja.__version__)"
python3 -c "import packaging; print('packaging ok')"

echo "=== 2. 设置 CUDA 环境 ==="
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -2
echo "TORCH_CUDA_ARCH_LIST will be auto-detected"

echo "=== 3. venv mamba_ssm before ==="
python3 -c "import mamba_ssm; print('before:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 4. 从本地路径编译（--no-build-isolation）==="
cd /mnt/e/mamba_official
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation . 2>&1 | tail -80

echo "=== 5. venv mamba_ssm after ==="
python3 -c "import mamba_ssm; print('after:', mamba_ssm.__version__, '|', mamba_ssm.__file__)"

echo "=== 6. try Mamba3 ==="
python3 -c "
from mamba_ssm import Mamba3
print('✅ Mamba3 import OK')
m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16, is_mimo=False)
print('✅ 实例化成功, params:', sum(p.numel() for p in m.parameters()))
"

echo "=== DONE ==="
