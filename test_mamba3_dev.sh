#!/bin/bash
# 验证 PYTHONPATH 开发模式：用 /mnt/e/mamba_official 源码跑 Mamba3，不编译
source "$HOME/mamba3_venv/bin/activate"
cd /mnt/e/three_chain_v3
export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH

echo "=== 1. mamba_ssm 来源 ==="
python3 -c "import mamba_ssm; print('path:', mamba_ssm.__file__); print('version:', mamba_ssm.__version__)"

echo "=== 2. Mamba3 import + 实例化 ==="
python3 -c "
from mamba_ssm import Mamba3
import torch
m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16, is_mimo=False).cuda()
print('✅ 实例化成功, params:', sum(p.numel() for p in m.parameters()))
"

echo "=== 3. forward 测试（GPU）==="
python3 -c "
import torch
from mamba_ssm import Mamba3
torch.manual_seed(0)
m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16, is_mimo=False).cuda()
x = torch.randn(2, 128, 64, device='cuda')
import time
# warmup
y = m(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    y = m(x)
torch.cuda.synchronize()
dt = (time.time() - t0) / 5
print(f'✅ forward OK, shape={y.shape}, time={dt*1000:.1f}ms')
"

echo "=== 4. backward 测试 ==="
python3 -c "
import torch
from mamba_ssm import Mamba3
torch.manual_seed(0)
m = Mamba3(d_model=64, d_state=64, headdim=32, chunk_size=16, is_mimo=False).cuda()
x = torch.randn(2, 128, 64, device='cuda', requires_grad=True)
y = m(x)
loss = y.sum()
loss.backward()
print(f'✅ backward OK, grad norm: {x.grad.norm().item():.4f}')
"

echo "=== 5. 对比 Mamba2 速度（同 d_model）==="
python3 -c "
import torch, time
# 临时取消 PYTHONPATH，用 .local 的 Mamba2
import sys
sys.path = [p for p in sys.path if 'mamba_official' not in p]
import importlib
if 'mamba_ssm' in sys.modules:
    del sys.modules['mamba_ssm']
from mamba_ssm import Mamba2
m2 = Mamba2(d_model=64, d_state=64, headdim=32, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 64, device='cuda')
y = m2(x); torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    y = m2(x)
torch.cuda.synchronize()
dt = (time.time() - t0) / 5
print(f'Mamba2 forward: {dt*1000:.1f}ms, params={sum(p.numel() for p in m2.parameters())}')
"

echo "=== DONE ==="
