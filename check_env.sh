#!/bin/bash
echo '=== Python ==='
python3 --version
which python3
echo '=== existing mamba_ssm ==='
pip show mamba_ssm 2>/dev/null | head -5 || echo 'not installed in user space'
echo '=== CUDA toolkit (nvcc) ==='
nvcc --version 2>/dev/null || echo 'NO_NVCC (need cuda-toolkit for triton compile)'
echo '=== gcc/g++ ==='
gcc --version | head -1
g++ --version | head -1
echo '=== torch CUDA ==='
python3 -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA version:", torch.version.cuda)'
echo '=== venv module ==='
python3 -c 'import venv; print("venv ok")'
echo '=== cuda homes ==='
ls /usr/local/ | grep -i cuda || echo 'no /usr/local/cuda'
echo '=== existing mamba_ssm location ==='
python3 -c 'import mamba_ssm; print("mamba_ssm path:", mamba_ssm.__file__); print("mamba_ssm version:", mamba_ssm.__version__)' 2>&1 | head -3
