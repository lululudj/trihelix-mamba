"""
A 方案: 本地训练 max_T=1024 baseline (3.26M), 然后评估 T=150/300/500
训练 T=100, 评估超长外推
"""
import subprocess
import sys
import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CONFIG = "configs/matched_mamba2_extreme.yaml"
OUT_DIR = "results/run_extreme_A_baseline_maxT1024"

print("=" * 70)
print("A 方案: 训练 max_T=1024 baseline (3.26M) + 评估 T=150/300/500")
print("=" * 70)

# Step 1: 训练 500 步
print("\n[1] 训练 max_T=1024 baseline (500 步) ...")
cmd_train = [
    sys.executable, "-u", "train.py",
    "--model", "three_chain_mamba2",
    "--config", CONFIG,
    "--max_steps", "500",
    "--batch_size", "4",
    "--seed", "0",
    "--data_root", "./data_cache",
    "--out_dir", OUT_DIR,
]
r = subprocess.run(cmd_train, cwd=r"e:\three_chain_v3")
if r.returncode != 0:
    print(f"✗ 训练失败 rc={r.returncode}")
    sys.exit(1)
print("  ✓ 训练完成")

# Step 2: 评估 T=150, T=300, T=500
datasets = [
    ("T150", r"e:\three_chain_v3\data\ood_T150"),
    ("T300", r"e:\three_chain_v3\data\ood_T300"),
    ("T500", r"e:\three_chain_v3\data\ood_T500"),
]

print("\n[2] 评估超长 OOD 外推 ...")
for name, data_root in datasets:
    out = f"{OUT_DIR}/ood_metrics_{name}.json"
    cmd_eval = [
        sys.executable, "-u", "eval_ood.py",
        "--checkpoint", f"{OUT_DIR}/final.pt",
        "--config", CONFIG,
        "--data_root", data_root,
        "--batch_size", "4",
        "--seed", "0",
        "--out", out,
    ]
    r = subprocess.run(cmd_eval, cwd=r"e:\three_chain_v3")
    if r.returncode == 0:
        print(f"  ✓ {name} 评估完成 -> {out}")
    else:
        print(f"  ✗ {name} 评估失败 rc={r.returncode}")

print("\n" + "=" * 70)
print("A 方案完成")
print("=" * 70)
