import sys, torch
sys.path.insert(0, '/mnt/e/three_chain_v3')
from models.common import HeteroMamba
from models.baselines import SingleChain, TransformerBaseline

# Tri-Helix: n_layer=5, d_state=17
print('=== Tri-Helix ===')
m1 = HeteroMamba(d_model=256, n_layer=5, d_state=17, n_experts=3, expert_type='triple', base_pair=True)
p1 = sum(p.numel() for p in m1.parameters())
print(f'Params: {p1:,} ({p1/1e6:.2f}M)')

# Single Mamba: n_layer=8
print('=== Single Mamba ===')
m2 = SingleChain(d_model=256, n_layer=8, d_state=16)
p2 = sum(p.numel() for p in m2.parameters())
print(f'Params: {p2:,} ({p2/1e6:.2f}M)')

# Transformer: n_layer=6
print('=== Transformer ===')
m3 = TransformerBaseline(d_model=256, n_layer=6, n_head=8, ffn_mult=3)
p3 = sum(p.numel() for p in m3.parameters())
print(f'Params: {p3:,} ({p3/1e6:.2f}M)')

print(f'DIFF: TH-SM={abs(p1-p2):,} TH-TF={abs(p1-p3):,} SM-TF={abs(p2-p3):,}')
