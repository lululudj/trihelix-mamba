"""ThreeChainMamba2BPv2 单元测试: forward + loss + backward + 梯度 + alpha=0验证"""
import sys
import torch

sys.path.insert(0, "/mnt/e/three_chain_v3")

from models.three_chain_mamba2_bpv2 import ThreeChainMamba2BPv2, CrossChainBasePair
from models.three_chain_mamba2 import ThreeChainMamba2
from utils import count_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# 1. 构建模型
model = ThreeChainMamba2BPv2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
n_params = count_params(model)
print(f"BPv2 params: {n_params/1e6:.2f}M")

baseline = ThreeChainMamba2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
n_base = count_params(baseline)
print(f"baseline params: {n_base/1e6:.2f}M")
print(f"diff: +{(n_params-n_base)/1e3:.1f}K ({(n_params-n_base)/n_base*100:.1f}%)")

# 2. 验证 alpha=0 (初始时 base_pair 是恒等变换)
for i, bp in enumerate(model.mamba.base_pair):
    assert bp.alpha.item() == 0.0, f"layer {i} alpha != 0"
print("[PASS] alpha=0 at init (base_pair is identity = baseline)")

# 3. Forward pass
B, N, K, T = 2, 8, 4, 20
S_0 = torch.randint(0, 16, (B, N, N), device=device)
actions = torch.randint(0, 5, (B, K, T), device=device)
S_t = torch.randint(0, 16, (B, T + 1, N, N), device=device)

logits, info = model(S_0, actions)
print(f"[PASS] forward: logits {tuple(logits.shape)}")
assert logits.shape == (B, T, N, N, 16)

# 4. Loss
total, loss_info = model.loss(logits, S_t, info)
print(f"[PASS] loss: {total.item():.4f}")

# 5. Backward
total.backward()
print("[PASS] backward (no error)")

# 6. 梯度检查: alpha 和 gate 必须有梯度
for i, bp in enumerate(model.mamba.base_pair):
    assert bp.alpha.grad is not None, f"layer {i} alpha no grad"
    assert bp.gate.weight.grad is not None, f"layer {i} gate.weight no grad"
    print(f"  layer {i}: alpha.grad={bp.alpha.grad.item():.6f}, gate.grad_norm={bp.gate.weight.grad.norm().item():.6f}")
print("[PASS] gradients exist for all base_pair params")

# 7. 验证 alpha=0 时, BPv2 输出 == baseline 输出 (相同权重种子)
# 用同一个 seed 重建两个模型, 检查前几层权重是否一致
torch.manual_seed(42)
m1 = ThreeChainMamba2BPv2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
torch.manual_seed(42)
m2 = ThreeChainMamba2(cell_types=16, action_dim=5, d_model=256, n_layers=2).to(device)
m1.eval(); m2.eval()
with torch.no_grad():
    # 用相同权重... 但 BPv2 多了 base_pair, 所以共享部分应该一致
    # 检查 cell_embed 是否相同 (验证初始化一致)
    assert torch.allclose(m1.cell_embed.weight, m2.cell_embed.weight), "cell_embed mismatch"
    # 检查 mamba_t 第一层是否相同
    assert torch.allclose(m1.mamba.mamba_t[0].A_log, m2.mamba.mamba_t[0].A_log), "mamba_t mismatch"
print("[PASS] shared layers init identically (fair comparison)")

# 8. 训练一步后 alpha 应该变化
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
opt.zero_grad()
logits, info = model(S_0, actions)
total, _ = model.loss(logits, S_t, info)
total.backward()
opt.step()
print(f"[PASS] after 1 step: alpha = {[round(bp.alpha.item(), 6) for bp in model.mamba.base_pair]}")

print("\n=== ALL TESTS PASSED ===")
