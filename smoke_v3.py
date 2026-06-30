import sys; sys.path.insert(0, '.')
from models.three_chain import ThreeChain
import torch

m = ThreeChain()
s = torch.randint(0, 12, (2, 8, 8))
a = torch.randint(0, 5, (2, 4, 100))
logits, info = m(s, a)
print('logits:', logits.shape)
print('aux keys:', list(info['aux'].keys()))

# Test loss
S_t = torch.randint(0, 12, (2, 101, 8, 8))
total, loss_info = m.loss(logits, S_t, info)
print('total_loss:', total.item())
print('loss_info keys:', list(loss_info.keys()))
print('OK v3 smkoke test passed!')
