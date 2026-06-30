import sys, torch
sys.path.insert(0, ".")
from models.baselines import SingleChain, TransformerBaseline

print("=== WSL SingleChain (real mamba_ssm) ===")
for nl in [9, 18, 27, 36, 37, 38]:
    try:
        m = SingleChain(cell_types=16, action_dim=5, d_model=256, n_layers=nl)
        p = sum(p.numel() for p in m.parameters())
        print(f"  n_layers={nl:3d}  params={p:,}")
    except Exception as e:
        print(f"  n_layers={nl:3d}  ERROR: {e}")

print()
print("=== WSL Transformer ===")
for nl in [5, 8, 10, 12, 13, 14]:
    try:
        m = TransformerBaseline(cell_types=16, action_dim=5, d_model=256, n_layers=nl, n_heads=4)
        p = sum(p.numel() for p in m.parameters())
        print(f"  n_layers={nl:3d}  params={p:,}")
    except Exception as e:
        print(f"  n_layers={nl:3d}  ERROR: {e}")
