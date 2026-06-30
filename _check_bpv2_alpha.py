"""检查 BPv2 训练后 alpha 学到了多少 + 测试更深模型/更长OOD"""
import sys
import torch

sys.path.insert(0, "/mnt/e/three_chain_v3")

from models.three_chain_mamba2_bpv2 import ThreeChainMamba2BPv2
from utils import set_seed, get_device, count_params, load_config
from data.dataset import make_loaders

cfg = load_config("configs/matched_mamba2.yaml")
device = get_device()
set_seed(42)

# 1. 构建 BPv2, 快速训练 200 步, 检查 alpha
print("=== BPv2 alpha 激活检查 (200步快速训练) ===")
model = ThreeChainMamba2BPv2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
print(f"初始 alpha: {[round(bp.alpha.item(), 4) for bp in model.mamba.base_pair]}")

loaders = make_loaders("./data_split", batch_size=2, seed=42)
train_loader = loaders["train"]
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

from probe_mamba2_brain import train_steps
train_steps(model, train_loader, opt, 200, device,
            cfg["train"]["lr"], cfg["train"]["warmup_steps"], 0.3)

print(f"\n200步后 alpha: {[round(bp.alpha.item(), 4) for bp in model.mamba.base_pair]}")
print(f"gate 权重范数: {[round(bp.gate.weight.norm().item(), 4) for bp in model.mamba.base_pair]}")

# 2. 试更深模型 (n_layers=4), 看横档是否在深层更有效
print("\n=== 深层 BPv2 (n_layers=4) 参数量 ===")
model_deep = ThreeChainMamba2BPv2(cell_types=16, action_dim=5, d_model=256, n_layers=4).to(device)
n_deep = count_params(model_deep)
print(f"n_layers=4: {n_deep/1e6:.2f}M params")
print(f"初始 alpha: {[round(bp.alpha.item(), 4) for bp in model_deep.mamba.base_pair]}")
