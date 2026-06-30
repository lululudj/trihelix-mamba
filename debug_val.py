"""调试 val changed_acc 不变的问题：检查模型在 val 上的预测分布。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from collections import Counter

from data.dataset import make_loaders
from utils import set_seed, get_device, load_config, build_model, load_checkpoint, cell_accuracy

cfg = load_config("configs/default.yaml")
set_seed(0)
device = get_device()

# 加载当前 best.pt（应该是 step 500 的版本，因为 val acc 不变）
state = torch.load("results/run_three_chain_seed0/best.pt", map_location=device, weights_only=False)
print(f"best.pt step={state.get('step')} best_val={state.get('best_val')}")

model = build_model("three_chain", cfg).to(device)
model.load_state_dict(state["model"])
model.eval()

loaders = make_loaders("./data_split", batch_size=8, seed=0)
val_loader = loaders["val"]

# 检查第一个 val batch 的预测
batch = next(iter(val_loader))
S_0 = batch["S_0"].to(device)
actions = batch["actions"].to(device)
S_t = batch["S_t"].to(device)

with torch.no_grad():
    logits, info = model(S_0, actions)

pred = logits.argmax(dim=-1)  # (B, T, N, N)
target = S_t[:, 1:]
print(f"\npred shape: {pred.shape}")
print(f"target shape: {target.shape}")

# 看 pred 分布
pred_counts = Counter(pred.flatten().tolist())
target_counts = Counter(target.flatten().tolist())
print(f"\npred 分布（前10）: {dict(sorted(pred_counts.items(), key=lambda x: -x[1])[:10])}")
print(f"target 分布（前10）: {dict(sorted(target_counts.items(), key=lambda x: -x[1])[:10])}")

# 变化 cell 准确率
changed = (target != S_0.unsqueeze(1))
print(f"\n变化 cell 总数: {changed.sum().item()}")
print(f"变化 cell 中 pred==target: {(pred[changed] == target[changed]).sum().item()}")
print(f"changed_acc: {(pred[changed] == target[changed]).float().mean().item():.6f}")

# 看模型是否在所有 step 都预测同一个值
print(f"\n每个 t 步的 pred 唯一值数:")
for t in range(pred.shape[1]):
    n_unique = len(set(pred[:, t].flatten().tolist()))
    print(f"  t={t+1}: {n_unique} unique values")

# 看 pred == S_0 的比例（"全猜不变"的比例）
pred_eq_init = (pred == S_0.unsqueeze(1)).float().mean().item()
print(f"\npred==S_0 比例: {pred_eq_init:.4f}")

# 跟 v1 best.pt 对比（如果存在）
v1_ckpt = Path("results_v1/run_three_chain_seed0/best.pt")
if v1_ckpt.exists():
    print(f"\n=== 对比 v1 best.pt ===")
    state_v1 = torch.load(v1_ckpt, map_location=device, weights_only=False)
    model_v1 = build_model("three_chain", cfg).to(device)
    # v1 是 cell_types=12，现在是 16，可能加载失败
    try:
        model_v1.load_state_dict(state_v1["model"])
        model_v1.eval()
        with torch.no_grad():
            logits_v1, _ = model_v1(S_0, actions)
        pred_v1 = logits_v1.argmax(dim=-1)
        ch_acc_v1 = (pred_v1[changed] == target[changed]).float().mean().item()
        print(f"v1 model changed_acc: {ch_acc_v1:.6f}")
        print(f"v2 model changed_acc: {(pred[changed] == target[changed]).float().mean().item():.6f}")
    except Exception as e:
        print(f"v1 加载失败（预期，cell_types 不同）: {e}")
