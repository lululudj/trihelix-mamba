"""最小测试: BP 模型能否 forward + backward 一次。
定位 CUDA driver error 是模型问题还是环境问题。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from utils import set_seed, load_config, build_model

set_seed(42)
device = torch.device("cuda")
cfg = load_config("configs/matched_mamba2.yaml")

print("=== 构造 BP 模型 ===")
model = build_model("three_chain_mamba2_bp", cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f}M")

# 假数据
B, N, K, T = 2, 8, 3, 50
S_0 = torch.randint(0, 16, (B, N, N), device=device)
actions = torch.randint(0, 5, (B, K, T), device=device)
S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device)

print("\n=== forward 测试 ===")
try:
    logits, info = model(S_0, actions)
    print(f"✅ forward 成功: logits {tuple(logits.shape)}")
    print(f"   logits 范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"   logits 是否全0: {(logits == 0).all().item()}")
except Exception as e:
    print(f"❌ forward 失败: {e}")
    sys.exit(1)

print("\n=== loss 测试 ===")
try:
    loss, loss_info = model.loss(logits, S_t, info, aux_weight=0.3)
    print(f"✅ loss 成功: {loss.item():.4f}")
    print(f"   loss_info keys: {list(loss_info.keys())}")
    print(f"   entropy: {loss_info.get('entropy', 'N/A')}")
except Exception as e:
    print(f"❌ loss 失败: {e}")
    sys.exit(1)

print("\n=== backward 测试 ===")
try:
    loss.backward()
    print("✅ backward 成功")
    # 检查梯度
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"   有梯度的参数: {n_grad}/{n_total}")
    # 检查是否有 NaN 梯度
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print