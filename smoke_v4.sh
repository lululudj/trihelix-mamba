cd /mnt/e/three_chain_v3
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from models.three_chain import ThreeChain

device = torch.device('cuda')
print('Building HeteroMamba v4...')
m = ThreeChain().to(device)
total = sum(p.numel() for p in m.parameters())
print(f'Total params: {total/1e6:.2f}M')

print('Forward pass...')
s = torch.randint(0, 12, (2, 8, 8)).to(device)
a = torch.randint(0, 5, (2, 4, 100)).to(device)
logits, info = m(s, a)
print(f'logits: {logits.shape}')
print(f'aux keys: {list(info[\"aux\"].keys())}')

print('Loss...')
S_t = torch.randint(0, 12, (2, 101, 8, 8)).to(device)
t, li = m.loss(logits, S_t, info)
print(f'total_loss: {t.item():.4f}')
print(f'loss keys: {sorted(li.keys())}')
print('V4 smoke test PASSED!')
"
