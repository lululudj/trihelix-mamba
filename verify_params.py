import sys, torch
sys.path.insert(0, "/mnt/e/three_chain_v3")
from models.common import HeteroMamba, FusedMambaBlock
from models.baselines import SingleChain, TransformerBaseline

# Tri-Helix: n_layer=5, d_state=17
print("=== Tri-Helix (n_layer=5, d_state=17, d_model=256) ===")
m1 = HeteroMamba(d_model=256, n_layer=5, d_state=17, n_experts=3, expert_type="triple", base_pair=True)
p1 = sum(p.numel() for p in m1.parameters())
print(f"Params: {p1:,} ({p1/1e6:.2f}M)")

# Single Mamba: n_layer=8
print("\n=== Single Mamba (n_layer=8, d_model=256, d_state=16) ===")
m2 = SingleChain(d_model=256, n_layer=8, d_state=16)
p2 = sum(p.numel() for p in m2.parameters())
print(f"Params: {p2:,} ({p2/1e6:.2f}M)")

# Transformer: n_layer=6, ffn_mult=3
print("\n=== Transformer (n_layer=6, d_model=256, n_head=8, ffn_mult=3) ===")
m3 = TransformerBaseline(d_model=256, n_layer=6, n_head=8, ffn_mult=3)
p3 = sum(p.numel() for p in m3.parameters())
print(f"Params: {p3:,} ({p3/1e6:.2f}M)")

print(f"\n=== DIFF ===")
print(f"Tri-Helix vs Single: {abs(p1-p2):,} ({abs(p1-p2)/p1*100:.2f}%)")
print(f"Tri-Helix vs Transformer: {abs(p1-p3):,} ({abs(p1-p3)/p1*100:.2f}%)")
print(f"Single vs Transformer: {abs(p2-p3):,} ({abs(p2-p3)/p2*100:.2f}%)")
