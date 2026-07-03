#!/bin/bash
# 加 warmup 重测：区分 Triton JIT 编译时间和实际运行时间
source "$HOME/mamba3_venv/bin/activate"
cd /mnt/e/three_chain_v3
export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH

python3 << 'EOF'
import torch
import time
from models.three_chain_mamba3 import ThreeChainMamba3

torch.manual_seed(0)
m = ThreeChainMamba3(cell_types=16, action_dim=5, d_model=256, n_layers=2).cuda()
B, N, K, T = 2, 6, 4, 30
S_0 = torch.randint(0, 16, (B, N, N), device='cuda')
actions = torch.randint(0, 5, (B, K, T), device='cuda')
S_t = torch.randint(0, 16, (B, T+1, N, N), device='cuda')

print("=== 第 1 次 forward (含 Triton JIT 编译 + autotuner) ===")
torch.cuda.synchronize()
t0 = time.time()
logits, info = m(S_0, actions)
torch.cuda.synchronize()
print(f"  forward 1: {(time.time()-t0):.1f}s (含 JIT 编译)")

t0 = time.time()
total, loss_info = m.loss(logits, S_t, info, aux_weight=0.3)
total.backward()
torch.cuda.synchronize()
print(f"  backward 1: {(time.time()-t0):.1f}s (含 JIT 编译)")
m.zero_grad()

print()
print("=== 第 2 次 forward+backward (kernel 已缓存，纯运行时间) ===")
torch.cuda.synchronize()
t0 = time.time()
logits, info = m(S_0, actions)
torch.cuda.synchronize()
print(f"  forward 2: {(time.time()-t0)*1000:.1f}ms")

t0 = time.time()
total, loss_info = m.loss(logits, S_t, info, aux_weight=0.3)
total.backward()
torch.cuda.synchronize()
print(f"  backward 2: {(time.time()-t0)*1000:.1f}ms")
m.zero_grad()

print()
print("=== 第 3 次 (确认稳定) ===")
torch.cuda.synchronize()
t0 = time.time()
logits, info = m(S_0, actions)
total, loss_info = m.loss(logits, S_t, info, aux_weight=0.3)
total.backward()
torch.cuda.synchronize()
print(f"  forward+backward 3: {(time.time()-t0)*1000:.1f}ms")

print()
fwd_bwd_ms = (time.time()-t0)*1000
print(f"=== 100 步训练时间估计 (基于第 3 次) ===")
print(f"  首步 (含 JIT): ~350s (一次性)")
print(f"  后续 99 步每步: {fwd_bwd_ms/1000:.2f}s")
print(f"  总预估: {350 + 99*fwd_bwd_ms/1000:.0f}s")
print(f"  对照: mamba2=49s, mamba3_ref=1031s")
EOF
