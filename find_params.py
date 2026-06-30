import sys; sys.path.insert(0, "E:/three_chain_v3"); sys.stdout.reconfigure(encoding="utf-8")
import torch
from models.baselines import SingleChain, TransformerBaseline
TARGET = 4768962
print(f"Target: {TARGET:,}\n")
print("=== SingleChain ===")
for nl in range(8, 15):
    m = SingleChain(cell_types=16, action_dim=5, d_model=256, n_layers=nl)
    p = sum(p.numel() for p in m.parameters())
    diff = abs(p - TARGET) / TARGET * 100
    marker = " <-- BEST" if diff < 5 else ""
    print(f"  n_layers={nl:2d}  params={p:,}  diff={diff:.1f}%{marker}")
print("\n=== Transformer ===")
for nl in range(3, 8):
    m = TransformerBaseline(cell_types=16, action_dim=5, d_model=256, n_layers=nl, n_heads=4)
    p = sum(p.numel() for p in m.parameters())
    diff = abs(p - TARGET) / TARGET * 100
    marker = " <-- BEST" if diff < 5 else ""
    print(f"  n_layers={nl:2d}  params={p:,}  diff={diff:.1f}%{marker}")
