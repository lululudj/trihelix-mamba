"""绘制 OOD 长程外推对比图：三模型在 T=150 上的 changed_acc 曲线。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("results_wsl")
models = ["three_chain", "single_chain", "transformer"]
colors = {"three_chain": "#a570ff", "single_chain": "#5ac8fa", "transformer": "#ff7043"}
labels = {"three_chain": "Three-Chain (DNA-Mamba)",
          "single_chain": "Single-Chain",
          "transformer": "Transformer"}

data = {}
for m in models:
    p = root / f"run_{m}_seed0" / "ood_metrics.json"
    if p.exists():
        data[m] = json.loads(p.read_text(encoding="utf-8"))

if not data:
    print("未找到 OOD metrics，请先运行 eval_ood.py")
    exit(1)

# 图1：OOD changed_acc vs t (1..150)
fig, ax = plt.subplots(figsize=(10, 5.5))
for m in models:
    if m not in data:
        continue
    curve = data[m]["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, "-", color=colors[m], linewidth=2.5, label=labels[m])
    # 标注 t=100 (训练边界) 和 t=150 (OOD)
    if "100" in curve:
        ax.scatter([100], [curve["100"]], color=colors[m], s=80, zorder=5, edgecolors="black")
    if "150" in curve:
        ax.scatter([150], [curve["150"]], color=colors[m], s=80, zorder=5, edgecolors="black")

# 训练边界线
ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7, label="训练边界 t=100")
ax.axvspan(100, 150, alpha=0.1, color="red", label="OOD 外推区")

ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t（变化cell准确率）", fontsize=12)
ax.set_title("OOD 长程外推对比 — T=150（训练时 T=100）", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 155)
fig.tight_layout()
out1 = root / "ood_comparison_changed_acc.png"
fig.savefig(out1, dpi=140)
print(f"已绘图 {out1}")
plt.close(fig)

# 图2：OOD 衰减柱状图（t=100 vs t=150）
fig, ax = plt.subplots(figsize=(8, 5))
x_labels = []
vals_100 = []
vals_150 = []
for m in models:
    if m not in data:
        continue
    curve = data[m]["changed_acc_curve"]
    if "100" in curve and "150" in curve:
        x_labels.append(labels[m])
        vals_100.append(curve["100"])
        vals_150.append(curve["150"])

import numpy as np
x = np.arange(len(x_labels))
w = 0.35
ax.bar(x - w/2, vals_100, w, label="t=100 (训练长度)", color="#5ac8fa", edgecolor="black")
ax.bar(x + w/2, vals_150, w, label="t=150 (OOD 外推)", color="#ff7043", edgecolor="black")
for i, (v100, v150) in enumerate(zip(vals_100, vals_150)):
    ax.text(i - w/2, v100 + 0.005, f"{v100:.3f}", ha="center", fontsize=10)
    ax.text(i + w/2, v150 + 0.005, f"{v150:.3f}", ha="center", fontsize=10)
    # 衰减百分比
    decay = (v150 - v100) / v100 * 100 if v100 > 0 else 0
    ax.text(i, max(v100, v150) + 0.02, f"{decay:+.1f}%", ha="center",
            fontsize=11, fontweight="bold", color="red" if decay < 0 else "green")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel("Changed Acc", fontsize=12)
ax.set_title("OOD 外推衰减：t=100 vs t=150", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out2 = root / "ood_decay_bar.png"
fig.savefig(out2, dpi=140)
print(f"已绘图 {out2}")
plt.close(fig)

print("\nOOD 对比图绘制完成")
print("\n=== OOD 结果汇总 ===")
print(f"{'model':<22} {'acc@100':>10} {'acc@150':>10} {'decay':>10}")
print("-" * 55)
for m in models:
    if m not in data:
        continue
    curve = data[m]["changed_acc_curve"]
    a100 = curve.get("100", 0)
    a150 = curve.get("150", 0)
    decay = (a150 - a100) / a100 * 100 if a100 > 0 else 0
    print(f"{m:<22} {a100:>10.4f} {a150:>10.4f} {decay:>+9.1f}%")
