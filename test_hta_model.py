"""验证 ThreeChainMamba2(use_hta=True) 白嫖版能正常工作
[1] 参数量和 baseline 完全一致 (HTAParametrization 无可训练参数)
[2] forward + backward 能跑
[3] A 确实从 -exp(A_log) 变成 -heavy_tail(raw)
[4] checkpoint save/load 正常
"""
import torch
import torch.nn.utils.parametrize as P
import sys
sys.path.insert(0, '.')
from models.three_chain_mamba2 import ThreeChainMamba2, heavy_tail_activation

torch.manual_seed(0)
device = 'cuda'

print("=== [1] 构造 baseline 和 hta 版, 比较参数量 ===")
m_base = ThreeChainMamba2(d_model=256, n_layers=2, use_hta=False).to(device)
m_hta = ThreeChainMamba2(d_model=256, n_layers=2, use_hta=True).to(device)

n_base = sum(p.numel() for p in m_base.parameters())
n_hta = sum(p.numel() for p in m_hta.parameters())
print(f"  baseline params: {n_base:,}")
print(f"  hta params:       {n_hta:,}")
print(f"  差异: {n_hta - n_base} (应=0, HTAParametrization 无可训练参数)")
assert n_base == n_hta, "❌ 参数量不一致!"
print("  ✅ 参数量完全一致")

print()
print("=== [2] forward + backward ===")
B, N, K, T = 2, 6, 4, 30
S_0 = torch.randint(0, 16, (B, N, N), device=device)
actions = torch.randint(0, 5, (B, K, T), device=device)
S_t = torch.randint(0, 16, (B, T+1, N, N), device=device)

logits, info = m_hta(S_0, actions)
print(f"  forward OK, logits shape: {logits.shape}")
total, loss_info = m_hta.loss(logits, S_t, info, aux_weight=0.3)
total.backward()
print(f"  backward OK, loss: {total.item():.4f}, ch_acc: {loss_info.get('changed_acc', 'N/A')}")
m_hta.zero_grad()

print()
print("=== [3] 验证 A 确实用了 heavy_tail ===")
# 检查时间链第一个 Mamba2 是否被 parametrize
mamba_t0 = m_hta.mamba.mamba_t[0]
mamba_s0 = m_hta.mamba.mamba_s_row[0].mamba
mamba_c0 = m_hta.mamba.mamba_c[0]
for name, mod in [("时间链 mamba_t[0]", mamba_t0),
                   ("空间链 mamba_s_row[0]", mamba_s0),
                   ("因果链 mamba_c[0]", mamba_c0)]:
    is_param = P.is_parametrized(mod, 'A_log')
    print(f"  {name}: A_log parametrized = {is_param}")
    assert is_param, f"❌ {name} 未被 parametrize!"

# 确认 baseline 未被 parametrize
mamba_t0_base = m_base.mamba.mamba_t[0]
is_param_base = P.is_parametrized(mamba_t0_base, 'A_log')
print(f"  baseline mamba_t[0]: A_log parametrized = {is_param_base} (应为 False)")
assert not is_param_base, "❌ baseline 被意外 parametrize!"
print("  ✅ 三链全部 patch, baseline 未受影响")

print()
print("=== [4] checkpoint save/load ===")
import tempfile, os
ckpt_path = tempfile.mktemp(suffix='.pt')
torch.save({'model': m_hta.state_dict(), 'step': 100}, ckpt_path)

m_hta2 = ThreeChainMamba2(d_model=256, n_layers=2, use_hta=True).to(device)
state = torch.load(ckpt_path, weights_only=True)
m_hta2.load_state_dict(state['model'])
os.unlink(ckpt_path)
print(f"  save/load OK, step={state['step']}")

# 验证 load 后 forward 一致
logits2, _ = m_hta2(S_0, actions)
diff = (logits - logits2).abs().max().item()
print(f"  load 后 forward 差异: {diff:.2e} (应 < 1e-5)")
assert diff < 1e-5, "❌ load 后结果不一致!"
print("  ✅ checkpoint 正常")

print()
print("=" * 60)
print(" 🎉 白嫖版 ThreeChainMamba2HTA 全部验证通过!")
print("    - 参数量不变, baseline 不受影响")
print("    - 三链全部 patch (A: exp→heavy_tail)")
print("    - forward/backward/save/load 全通")
print("=" * 60)
