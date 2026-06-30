"""绘制三模型 changed_acc / acc 长程衰减对比图。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("results")
models = ["three_chain", "single_chain", "transformer"]
colors = {"three_chain": "#a570ff", "single_chain": "#5ac8fa", "transformer": "#ff7043"}
labels = {"three_chain": "Three-Chain (DNA-Mamba)", "single_chain": "Single-Chain", "transformer": "Transformer"}

data = {}
for m in models:
    p = root / f"run_{m}_seed0" / "metrics.json"
    data[m] = json.loads(p.read_text(encoding="utf-8"))

# 图1：changed_acc vs t（核心区分指标）
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in models:
    curve = data[m]["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, "o-", color=colors[m], linewidth=2.5, markersize=7, label=labels[m])
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t（变化cell准确率）", fontsize=12)
ax.set_title("三模型长程衰减对比 — Changed Cell Accuracy", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.25, 0.5)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 30, 60, 100])
ax.set_xticklabels(["1", "5", "10", "30", "60", "100"])
fig.tight_layout()
out1 = root / "comparison_changed_acc.png"
fig.savefig(out1, dpi=140)
print(f"已绘图 {out1}")
plt.close(fig)

# 图2：acc vs t（全cell准确率）
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in models:
    curve = data[m]["acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, "o-", color=colors[m], linewidth=2.5, markersize=7, label=labels[m])
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Acc@t（全cell准确率）", fontsize=12)
ax.set_title("三模型长程衰减对比 — All Cell Accuracy", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 0.75)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 30, 60, 100])
ax.set_xticklabels(["1", "5", "10", "30", "60", "100"])
fig.tight_layout()
out2 = root / "comparison_acc.png"
fig.savefig(out2, dpi=140)
print(f"已绘图 {out2}")
plt.close(fig)

# 图3：正交性热力图（three_chain）— 对称填充
orth = data["three_chain"]["orthogonality"]
keys = ["h_s", "h_c", "h_t"]
labels3 = ["空间轴 h_s", "因果轴 h_c", "时间轴 h_t"]
def get_orth(k1, k2):
    if f"{k1}_{k2}" in orth:
        return orth[f"{k1}_{k2}"]
    return orth[f"{k2}_{k1}"]
mat = [[get_orth(k1, k2) for k2 in keys] for k1 in keys]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels3, fontsize=11); ax.set_yticklabels(labels3, fontsize=11)
for i in range(3):
    for j in range(3):
        v = mat[i][j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black", fontsize=11, fontweight="bold")
ax.set_title("三轴隐状态正交性（余弦相似度）", fontsize=13, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out3 = root / "orthogonality_heatmap.png"
fig.savefig(out3, dpi=140)
print(f"已绘图 {out3}")
plt.close(fig)

# 图4：参数效率 vs 精度散点图
fig, ax = plt.subplots(figsize=(8, 5.5))
for m in models:
    par = data[m]["params"] / 1e6
    acc = data[m]["changed_acc"]
    ax.scatter(par, acc, s=200, color=colors[m], edgecolors="black", linewidth=1.5, zorder=3)
    ax.annotate(labels[m], (par, acc), xytext=(8, 8), textcoords="offset points", fontsize=11)
ax.set_xlabel("参数量 (M)", fontsize=12)
ax.set_ylabel("Changed Acc（变化cell准确率）", fontsize=12)
ax.set_title("参数量 vs 精度散点图", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.8, 4.0)
ax.set_ylim(0.35, 0.47)
fig.tight_layout()
out4 = root / "param_vs_acc.png"
fig.savefig(out4, dpi=140)
print(f"已绘图 {out4}")
plt.close(fig)

print("\n所有对比图绘制完成")
