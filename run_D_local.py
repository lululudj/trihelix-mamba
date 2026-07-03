"""
D 方案: 噪声/遮蔽鲁棒性评估 (本地 baseline 3.26M)
用 ood_T200_noise 和 ood_T250_mask 数据评估现有 checkpoint
"""
import subprocess
import sys
from pathlib import Path

CKPT = r"e:\three_chain_v3\results\run_three_chain_mamba2_100step_m3subset_seed0\final.pt"
CONFIG = "configs/matched_mamba2.yaml"

datasets = [
    ("T200_noise", r"e:\three_chain_v3\data\ood_T200_noise"),
    ("T250_mask",  r"e:\three_chain_v3\data\ood_T250_mask"),
    # 顺便也评估 T150 做对照
    ("T150_baseline", r"e:\three_chain_v3\data\ood_T150"),
]

print("=" * 70)
print("D 方案: 噪声/遮蔽鲁棒性评估 (本地 baseline 3.26M)")
print("=" * 70)

for name, data_root in datasets:
    out = f"results/extreme_D_{name}.json"
    print(f"\n>>> 评估 {name} ...")
    cmd = [
        sys.executable, "-u", "eval_ood.py",
        "--checkpoint", CKPT,
        "--config", CONFIG,
        "--data_root", data_root,
        "--batch_size", "4",
        "--seed", "0",
        "--out", out,
    ]
    r = subprocess.run(cmd, cwd=r"e:\three_chain_v3")
    if r.returncode == 0:
        print(f"  ✓ {name} 完成 -> {out}")
    else:
        print(f"  ✗ {name} 失败 rc={r.returncode}")

print("\n" + "=" * 70)
print("D 方案完成")
print("=" * 70)
