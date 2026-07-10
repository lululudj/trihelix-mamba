"""本地测试BP模块梯度流: 验证v2.1加法修复是否让梯度流过bp1_gate和bp2_reset"""
import torch
import torch.nn as nn
import sys, os

# 直接导入BP模块 (需要先处理依赖)
# three_chain_mamba3.py 导入了 mamba3_ref, 而后者需要 mamba_ssm
# 但 PairwiseBasePairMamba3 本身只需要 torch
# 让我们直接从文件中提取 BP 类的代码

# 先尝试直接导入
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.three_chain_mamba3 import PairwiseBasePairMamba3
    print("✅ 直接导入成功")
except ImportError as e:
    print(f"⚠️ 直接导入失败: {e}")
    print("尝试隔离测试BP模块...")

    # 手动构建与BP模块相同的网络结构来测试梯度
    D = 64

    # 复制BP模块的关键网络结构
    st_s2t_gate = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    tc_c2t_reset = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    tc_t2c_phase = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    cs_c2s_gamma = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))

    # 零初始化末层
    for mod in [st_s2t_gate, tc_c2t_reset, tc_t2c_phase, cs_c2s_gamma]:
        last_lin = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_lin.weight)
        nn.init.zeros_(last_lin.bias)

    # 测试输入
    B, T, N2 = 2, 10, 16
    x_t = torch.randn(B, T, N2, D)
    s_diff = torch.randn(B, D)
    g_c = torch.randn(B, D)

    # v2.1: 加法
    s_gate = st_s2t_gate(s_diff)          # (B, D) initial 0
    c_energy = torch.norm(g_c, dim=-1, keepdim=True) / (D ** 0.5)
    reset_gate = tc_c2t_reset(g_c) * c_energy.sigmoid()  # (B, D) initial 0

    reset_b = reset_gate.view(B, 1, 1, D)
    sgate_b = s_gate.view(B, 1, 1, D)
    x_t_new = x_t + reset_b * x_t + sgate_b * x_t  # v2.1 加法

    loss = x_t_new.sum()
    loss.backward()

    print("\n=== v2.1 加法: 梯度检查 ===")
    for name, net in [("st_s2t_gate (bp1_gate)", st_s2t_gate),
                       ("tc_c2t_reset (bp2_reset)", tc_c2t_reset),
                       ("tc_t2c_phase (对照)", tc_t2c_phase),
                       ("cs_c2s_gamma (对照)", cs_c2s_gamma)]:
        last_lin = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
        grad_norm = last_lin.weight.grad.norm().item()
        print(f"  {name}: grad_norm = {grad_norm:.6f}")

    # 对比 v2: 乘法
    print("\n=== v2 乘法: 梯度检查 (对照) ===")
    st_s2t_gate2 = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    tc_c2t_reset2 = nn.Sequential(nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, D))
    for mod in [st_s2t_gate2, tc_c2t_reset2]:
        last_lin = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_lin.weight)
        nn.init.zeros_(last_lin.bias)

    s_gate2 = st_s2t_gate2(s_diff)
    reset_gate2 = tc_c2t_reset2(g_c) * c_energy.sigmoid()
    reset_b2 = reset_gate2.view(B, 1, 1, D)
    sgate_b2 = s_gate2.view(B, 1, 1, D)
    x_t_new2 = x_t + reset_b2 * sgate_b2 * x_t  # v2 乘法

    loss2 = x_t_new2.sum()
    loss2.backward()

    for name, net in [("st_s2t_gate2 (bp1_gate)", st_s2t_gate2),
                       ("tc_c2t_reset2 (bp2_reset)", tc_c2t_reset2)]:
        last_lin = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
        grad_norm = last_lin.weight.grad.norm().item()
        print(f"  {name}: grad_norm = {grad_norm:.6f}")

    print("\n=== 结论 ===")
    # v2.1
    g21_gate = [m for m in st_s2t_gate.modules() if isinstance(m, nn.Linear)][-1].weight.grad.norm().item()
    g21_reset = [m for m in tc_c2t_reset.modules() if isinstance(m, nn.Linear)][-1].weight.grad.norm().item()
    # v2
    g2_gate = [m for m in st_s2t_gate2.modules() if isinstance(m, nn.Linear)][-1].weight.grad.norm().item()
    g2_reset = [m for m in tc_c2t_reset2.modules() if isinstance(m, nn.Linear)][-1].weight.grad.norm().item()

    print(f"v2.1 bp1_gate 梯度: {g21_gate:.6f} (应非零)")
    print(f"v2.1 bp2_reset 梯度: {g21_reset:.6f} (应非零)")
    print(f"v2   bp1_gate 梯度: {g2_gate:.6f} (死锁=0)")
    print(f"v2   bp2_reset 梯度: {g2_reset:.6f} (死锁=0)")

    if g21_gate > 0.0001 and g21_reset > 0.0001:
        print("\n✅ v2.1加法修复成功: bp1_gate和bp2_reset梯度非零, 不再死锁!")
    else:
        print("\n❌ v2.1加法修复失败: 梯度仍为零")
    sys.exit(0)

# 如果直接导入成功, 测试完整的BP模块
print("\n=== 完整BP模块测试 ===")
bp = PairwiseBasePairMamba3(d_model=64)

ms_before = bp.modulation_strength()
print("训练前:")
for k, v in ms_before.items():
    print(f"  {k}: {v:.6f}")

B, T, N2, D = 2, 10, 16, 64
x_s = torch.randn(B, T, N2, D)
x_t = torch.randn(B, T, N2, D)
c_inject = torch.randn(B, T, 1, D)
x = torch.randn(B, T, N2, D)

x_s_new, x_t_new, c_inject_new = bp(x, x_s, x_t, c_inject)
loss = x_t_new.sum() + x_s_new.sum() + c_inject_new.sum()
loss.backward()

print("\n梯度:")
for name, net in [("st_s2t_gate (bp1_gate)", bp.st_s2t_gate),
                   ("tc_c2t_reset (bp2_reset)", bp.tc_c2t_reset),
                   ("tc_t2c_phase (bp2_phase)", bp.tc_t2c_phase),
                   ("cs_c2s_gamma (bp3_gamma)", bp.cs_c2s_gamma)]:
    last_lin = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    grad_norm = last_lin.weight.grad.norm().item()
    print(f"  {name}: grad_norm = {grad_norm:.6f}")

# 一步Adam
opt = torch.optim.Adam(bp.parameters(), lr=0.01)
opt.step()
ms_after = bp.modulation_strength()
print("\n一步Adam(lr=0.01)后:")
for k, v in ms_after.items():
    diff = v - ms_before[k]
    print(f"  {k}: {v:.6f} (delta={diff:+.6f})")
