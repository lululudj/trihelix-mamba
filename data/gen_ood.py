"""生成 OOD 长程外推测试数据：T=150（训练时 T=100）。

固定 N=8, K=8（与训练同配置），只 T 不同，验证长程外推能力。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gen_grid_world import generate_scenario

OUT = Path(__file__).parent / "ood_T150"
OUT.mkdir(parents=True, exist_ok=True)

N, K, T = 8, 8, 150
scenario_types = ["random", "goal_directed", "adversarial"]
p_transfers = [0.0, 0.3, 0.6]
n_per = 8  # 每组合 8 个 → 3*3*8 = 72 样本

seed_base = 100000  # 避免与训练数据 seed 冲突
i = 0
for st in scenario_types:
    for p in p_transfers:
        for j in range(n_per):
            seed = seed_base + i
            S_0, actions, trajectory, meta = generate_scenario(
                N, K, T, p_transfer=p, scenario_type=st, seed=seed
            )
            fname = f"scen_{N:02d}_{K}_{T:03d}_{int(p*10)}_{st}_{i:06d}.npz"
            np.savez(
                OUT / fname,
                S_0=S_0, actions=actions, S_t=trajectory,
                N=N, K=K, T=T, p_transfer=p, scenario_type=st,
            )
            i += 1

print(f"OOD 数据生成完成: {OUT}")
print(f"  样本数: {i}")
print(f"  N={N}, K={K}, T={T}（训练 T=100，OOD 外推 +50%）")
print(f"  scenario_types: {scenario_types}")
print(f"  p_transfers: {p_transfers}")
