#!/bin/bash
cd /mnt/e/three_chain_v3
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from models.three_chain import ThreeChain

device = torch.device('cuda')
print('Building V5 HeteroMamba + BasePair...')
m = ThreeChain().to(device)
total = sum(p.numel() for p in m.parameters())
print(f'Total params: {total/1e6:.2f}M')

print('Forward...')
s = torch.randint(0, 12, (2, 8, 8)).to(device)
a = torch.randint(0, 5, (2, 4, 100)).to(device)
logits, info = m(s, a)
print(f'logits: {logits.shape}')
print(f'aux keys: {sorted(info[\"aux\"].keys())}')
if 'bp_info' in info['aux']:
    bp = info['aux']['bp_info']
    print(f'bp_info: delta_mod={bp[\"delta_mod_mean\"]:.3f} reset={bp[\"reset_gate_mean\"]:.3f} s_gate={bp[\"s_gate_mean\"]:.3f}')

print('Loss...')
S_t = torch.randint(0, 12, (2, 101, 8, 8)).to(device)
t, li = m.loss(logits, S_t, info)
print(f'total_loss: {t.item():.4f}')
print('V5 smoke test PASSED!')
"
