"""
A 方案: 生成超长 OOD 数据 T=300 和 T=500
格式与 ood_T150 一致: N=8, K=8, 3 scenario_types × 3 p_transfers × 8 = 72 样本
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "data"))
from gen_grid_world import generate_scenario

def gen_ood(T, out_name):
    OUT = Path(__file__).parent / "data" / out_name
    OUT.mkdir(parents=True, exist_ok=True)
    N, K = 8, 8
    scenario_types = ["random", "goal_directed", "adversarial"]
    p_transfers = [0.0, 0.3, 0.6]
    n_per = 8
    seed_base = 200000 + T  # 避免冲突
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
    print(f"✓ {out_name}: {i} 样本, T={T} (训练 T=100, 外推 {T/100:.1f}×)")

if __name__ == "__main__":
    print("=== A 方案: 生成超长 OOD 数据 ===")
    gen_ood(300, "ood_T300")
    gen_ood(500, "ood_T500")
    print("\n完成。下一步: 用 max_T=1024 的 checkpoint 评估")
