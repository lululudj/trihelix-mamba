python3 -c "
import torch, sys
sys.path.insert(0, '.')
from models.three_chain import ThreeChain

device = torch.device('cuda')
m = ThreeChain().to(device)
s = torch.randint(0, 12, (2, 8, 8)).to(device)
a = torch.randint(0, 5, (2, 4, 100)).to(device)
logits, info = m(s, a)
print('logits:', logits.shape)
print('aux keys:', list(info['aux'].keys()))

S_t = torch.randint(0, 12, (2, 101, 8, 8)).to(device)
t, li = m.loss(logits, S_t, info)
print('total_loss:', t.item())
print('loss_info keys:', list(li.keys()))
print('OK v3 smoke test PASSED!')
"
