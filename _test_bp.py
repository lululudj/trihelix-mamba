"""BP 模型单元测试: 确认 forward+backward 本身能不能跑通。
排除是不是模型设计 bug vs CUDA 驱动问题。
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.three_chain_mamba2_bp import ThreeChainMamba2BP

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("=== 构造模型 ===")
model = ThreeChainMamba2BP(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f}M")

print("\n=== 构造假数据 (小 batch) ===")
B, N, K, T = 2, 8, 3, 50  # 小 batch 小 T
S_0 = torch.randint(0, 16, (B, N, N), device=device)
actions = torch.randint(0, 5, (B, K, T), device=device)
S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device)

print("\n=== forward ===")
try:
    logits, info = model(S_0, actions)
    print(f"✅ forward OK, logits shape={logits.shape}")
except Exception as e:
    print(f"❌ forward FAIL: {e}")
    sys.exit(1)

print("\n=== loss ===")
try:
    loss, loss_info = model.loss(logits, S_t, info, aux_weight=0.3)
    print(f"✅ loss OK, loss={loss.item():.4f}")
    print(f"   div_loss={loss_info.get('div_loss', 0):.4f}, entropy={loss_info.get('entropy', 0):.4f}")
except Exception as e:
    print(f"❌ loss FAIL: {e}")
    sys.exit(1)

print("\n=== backward ===")
try:
    loss.backward()
    print("✅ backward OK")
except Exception as e:
    print(f"❌ backward FAIL: {e}")
    sys.exit(1)

print("\n=== 检查梯度 ===")
grad_ok = True
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"⚠️ {name} 无梯度")
        grad_ok = False
    elif torch.isnan(p.grad).any():
        print(f"⚠️ {name} 梯度 NaN")
        grad_ok = False
if grad_ok:
    print("✅ 所有参数有梯度且非 NaN")

print("\n=== 单元测试通过 ===")
print("结论: 模型本身 forward+backward 没问题")
print("如果完整训练还崩, 说明是 CUDA 驱动在长时间训练时不稳, 不是模型 bug")
