"""快速参数对比: 找到和 ThreeChainMamba2 (3.26M) 匹配的 Transformer 配置"""
import sys
sys.path.insert(0, "/mnt/e/three_chain_v3")

import torch
from models.three_chain_mamba2 import ThreeChainMamba2
from models.baselines import TransformerBaseline

def count(m):
    return sum(p.numel() for p in m.parameters())

# ThreeChainMamba2 (baseline)
mamba = ThreeChainMamba2(cell_types=16, action_dim=5, d_model=256, n_layers=2)
mamba_n = count(mamba)
print(f"ThreeChainMamba2:  {mamba_n/1e6:.3f}M  (d=256, L=2)")

# Transformer 不同层数
print(f"\nTransformerBaseline (d=256, heads=4, ffn=1024):")
for L in [2, 3, 4, 5, 6]:
    try:
        t = TransformerBaseline(cell_types=16, action_dim=5, d_model=256, n_layers=L)
        n = count(t)
        ratio = n / mamba_n
        print(f"  L={L}: {n/1e6:.3f}M  ({ratio:.2f}x baseline)")
    except Exception as e:
        print(f"  L={L}: ERROR {e}")

# 也试 d_model=128 (更窄更深)
print(f"\nTransformerBaseline (d=128, heads=4, ffn=512):")
for L in [3, 4, 5, 6, 7, 8]:
    try:
        t = TransformerBaseline(cell_types=16, action_dim=5, d_model=128, n_layers=L)
        n = count(t)
        ratio = n / mamba_n
        print(f"  L={L}: {n/1e6:.3f}M  ({ratio:.2f}x baseline)")
    except Exception as e:
        print(f"  L={L}: ERROR {e}")
