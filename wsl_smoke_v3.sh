#!/bin/bash
cd "/mnt/e/????? (2)/three_chain_validation"
python3 -c '
import torch, sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from models.three_chain import ThreeChain
m = ThreeChain()
s = torch.randint(0, 12, (2, 8, 8))
a = torch.randint(0, 5, (2, 4, 100))
logits, info = m(s, a)
print("logits:", logits.shape)
print("aux keys:", list(info["aux"].keys()))
S_t = torch.randint(0, 12, (2, 101, 8, 8))
t, li = m.loss(logits, S_t, info)
print("total_loss:", t.item())
print("loss_info:", list(li.keys()))
print("OK v3 smoke test passed!")
'
