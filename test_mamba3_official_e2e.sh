#!/bin/bash
# 端到端测试: ThreeChainMamba3 (官方版) 完整 forward + backward + 参数量
source "$HOME/mamba3_venv/bin/activate"
cd /mnt/e/three_chain_v3
export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH

echo "=== 1. import + 实例化 (真实三链参数 d_model=256) ==="
python3 -c "
import torch
from models.three_chain_mamba3 import ThreeChainMamba3
m = ThreeChainMamba3(cell_types=16, action_dim=5, d_model=256, n_layers=2).cuda()
n_params = sum(p.numel() for p in m.parameters())
print(f'✅ 实例化成功, params: {n_params/1e6:.4f}M ({n_params})')
print(f'   (对照 mamba3_ref 版: 3.3284M, mamba2 版: 3.26M)')
"

echo ""
echo "=== 2. forward + backward + loss (真实输入 N=6 T=30 K=4) ==="
python3 -c "
import torch
from models.three_chain_mamba3 import ThreeChainMamba3
torch.manual_seed(0)
m = ThreeChainMamba3(cell_types=16, action_dim=5, d_model=256, n_layers=2).cuda()
B, N, K, T = 2, 6, 4, 30
S_0 = torch.randint(0, 16, (B, N, N), device='cuda')
actions = torch.randint(0, 5, (B, K, T), device='cuda')
S_t = torch.randint(0, 16, (B, T+1, N, N), device='cuda')

import time
torch.cuda.synchronize()
t0 = time.time()
logits, info = m(S_0, actions)
torch.cuda.synchronize()
fwd_ms = (time.time() - t0) * 1000
print(f'✅ forward OK, logits={logits.shape}, time={fwd_ms:.1f}ms')

total, loss_info = m.loss(logits, S_t, info, aux_weight=0.3)
print(f'✅ loss OK, loss={total.item():.4f}, keys={list(loss_info.keys())}')

t0 = time.time()
total.backward()
torch.cuda.synchronize()
bwd_ms = (time.time() - t0) * 1000
print(f'✅ backward OK, time={bwd_ms:.1f}ms')

# 检查梯度
grad_norms = []
for name, p in m.named_parameters():
    if p.grad is not None:
        grad_norms.append((name.split('.')[0], p.grad.norm().item()))
print(f'✅ 梯度检查: {len(grad_norms)} 个参数有梯度')

# 速度估计
print()
print('=== 3. 100 步训练时间估计 ===')
print(f'单步 forward+backward: {(fwd_ms+bwd_ms):.0f}ms')
print(f'100 步预估: {(fwd_ms+bwd_ms)*100/1000:.0f}s (mamba2 baseline: 49s, mamba3_ref: 1031s)')
"

echo ""
echo "=== 4. 检查和 mamba3_ref 参数量是否一致 ==="
python3 -c "
import torch
from models.three_chain_mamba3 import ThreeChainMamba3
m = ThreeChainMamba3(cell_types=16, action_dim=5, d_model=256, n_layers=2)
n = sum(p.numel() for p in m.parameters())
print(f'官方版: {n} ({n/1e6:.4f}M)')
print(f'ref 版: 3328400 (3.3284M)')
print(f'差异: {n - 3328400} ({(n-3328400)/3328400*100:+.2f}%)')
"

echo ""
echo "=== DONE ==="
