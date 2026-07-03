#!/bin/bash
echo '=== nvcc in cuda dirs ==='
ls /usr/local/cuda-12.6/bin/nvcc 2>/dev/null && echo "FOUND nvcc in cuda-12.6"
ls /usr/local/cuda/bin/nvcc 2>/dev/null && echo "FOUND nvcc in cuda (symlink)"
ls /usr/local/cuda-13.0/bin/nvcc 2>/dev/null && echo "FOUND nvcc in cuda-13.0"

echo '=== CUDA_HOME env ==='
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH" | tr ':' '\n' | grep -i cuda || echo 'no cuda in PATH'

echo '=== existing mamba_ssm 2.2.4 contents ==='
ls /home/administrator/.local/lib/python3.14/site-packages/mamba_ssm/ | head -20
echo '--- modules/ ---'
ls /home/administrator/.local/lib/python3.14/site-packages/mamba_ssm/modules/ 2>/dev/null
echo '--- __init__.py exports ---'
grep -E '^(from|import|__all__)' /home/administrator/.local/lib/python3.14/site-packages/mamba_ssm/__init__.py | head -20

echo '=== try import Mamba3 in existing 2.2.4 ==='
python3 -c "from mamba_ssm import Mamba3; print('Mamba3 ALREADY in 2.2.4!')" 2>&1 | head -5

echo '=== pip triton version ==='
python3 -c "import triton; print('triton:', triton.__version__)"
