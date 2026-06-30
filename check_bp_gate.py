"""检查 ThreeChainBP 的碱基对门控值。

关键诊断：门控初始为 0（tanh(0)=0 → 弱耦合），
训练后若仍为 ~0，说明 BP 层未生效，OOD 不变就解释得通了。
"""
import torch
from pathlib import Path

ckpt_path = Path("results_wsl/run_three_chain_bp_seed0/best.pt")
print(f"加载 checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
msd = state["model"]

print(f"\nstep: {state.get('step')}")
print(f"best_val: {state.get('best_val'):.4f}")

# 找所有 bp 相关参数
print("\n=== BP 模块参数 ===")
bp_keys = [k for k in msd.keys() if k.startswith("bp.")]
print(f"BP 参数总数: {len(bp_keys)}")

# 门控值（核心诊断）
print("\n=== 门控值（tanh 后）===")
for name in ["bp.gate_s", "bp.gate_c", "bp.gate_t"]:
    if name in msd:
        raw = msd[name].item()
        gated = torch.tanh(torch.tensor(raw)).item()
        print(f"  {name}: raw={raw:.4f}, tanh={gated:.4f}")
    else:
        print(f"  {name}: 不存在")

# 统计 BP 各子模块参数范数（看是否被训练）
print("\n=== BP cross-attention 参数范数（看是否被训练）===")
groups = {}
for k in bp_keys:
    if "gate" in k:
        continue
    prefix = ".".join(k.split(".")[:3])  # e.g. bp.s_from_c.q_proj
    if prefix not in groups:
        groups[prefix] = []
    groups[prefix].append(msd[k].float().norm().item())

for prefix, norms in sorted(groups.items()):
    avg = sum(norms) / len(norms)
    print(f"  {prefix}: avg_norm={avg:.4f} (n_params={len(norms)})")

# 对比：如果门控 tanh≈0，那么 cross-attention 的输出几乎不被注入
# 如果门控 tanh 较大（>0.1），说明耦合确实被激活了
print("\n=== 诊断结论 ===")
gates = []
for name in ["bp.gate_s", "bp.gate_c", "bp.gate_t"]:
    if name in msd:
        gates.append(torch.tanh(torch.tensor(msd[name].item())).item())
if gates:
    avg_gate = sum(gates) / len(gates)
    max_gate = max(abs(g) for g in gates)
    print(f"  平均门控值: {avg_gate:.4f}")
    print(f"  最大|门控|: {max_gate:.4f}")
    if max_gate < 0.05:
        print("  → 门控几乎为 0，BP 层基本未生效（耦合太弱，残差占主导）")
    elif max_gate < 0.2:
        print("  → 门控较小，BP 层生效但耦合较弱")
    else:
        print("  → 门控明显激活，BP 层生效")
