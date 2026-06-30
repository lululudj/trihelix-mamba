"""绘制 WSL 真实 Mamba 结果图：test + OOD 对比。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("results_wsl")
models = ["three_chain", "single_chain", "transformer"]
colors = {"three_chain": "#a570ff", "single_chain": "#5ac8fa", "transformer": "#ff7043"}
labels = {"three_chain": "Three-Chain (DNA-Mamba)",
          "single_chain": "Single-Chain",
          "transformer": "Transformer"}

# 加载 test + OOD metrics
test_data = {}
ood_data = {}
for m in models:
    tp = root / f"run_{m}_seed0" / "metrics.json"
    op = root / f"run_{m}_seed0" / "ood_metrics.json"
    if tp.exists():
        test_data[m] = json.loads(tp.read_text(encoding="utf-8"))
    if op.exists():
        ood_data[m] = json.loads(op.read_text(encoding="utf-8"))

print(f"test: {list(test_data.keys())}, ood: {list(ood_data.keys())}")

# === 图1：test 集 changed_acc vs t ===
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in models:
    if m not in test_data:
        continue
    curve = test_data[m]["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, "o-", color=colors[m], linewidth=2.5, markersize=7, label=labels[m])
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t", fontsize=12)
ax.set_title("WSL 真实 Mamba — test 集 changed_acc（T=100）", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 30, 60, 100])
ax.set_xticklabels(["1", "5", "10", "30", "60", "100"])
fig.tight_layout()
out1 = root / "wsl_test_changed_acc.png"
fig.savefig(out1, dpi=140)
print(f"已绘图 {out1}")
plt.close(fig)

# === 图2：OOD 长程外推 changed_acc vs t (1..150) ===
fig, ax = plt.subplots(figsize=(10, 5.5))
for m in models:
    if m not in ood_data:
        continue
    curve = ood_data[m]["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, "-", color=colors[m], linewidth=2.5, label=labels[m])
    if "100" in curve:
        ax.scatter([100], [curve["100"]], color=colors[m], s=80, zorder=5, edgecolors="black")
    if "150" in curve:
        ax.scatter([150], [curve["150"]], color=colors[m], s=80, zorder=5, edgecolors="black")
ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7, label="训练边界 t=100")
ax.axvspan(100, 150, alpha=0.1, color="red", label="OOD 外推区")
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t", fontsize=12)
ax.set_title("OOD 长程外推 — changed_acc（T=150，训练 T=100）", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 155)
fig.tight_layout()
out2 = root / "wsl_ood_changed_acc.png"
fig.savefig(out2, dpi=140)
print(f"已绘图 {out2}")
plt.close(fig)

# === 图3：OOD 衰减柱状图（t=100 vs t=150）===
fig, ax = plt.subplots(figsize=(8, 5))
x_labels = []
vals_100 = []
vals_150 = []
decays = []
for m in models:
    if m not in ood_data:
        continue
    curve = ood_data[m]["changed_acc_curve"]
    if "100" in curve and "150" in curve:
        x_labels.append(labels[m])
        vals_100.append(curve["100"])
        vals_150.append(curve["150"])
        decays.append((curve["150"] - curve["100"]) / curve["100"] * 100 if curve["100"] > 0 else 0)
x = np.arange(len(x_labels))
w = 0.35
ax.bar(x - w/2, vals_100, w, label="t=100 (训练长度)", color="#5ac8fa", edgecolor="black")
ax.bar(x + w/2, vals_150, w, label="t=150 (OOD 外推)", color="#ff7043", edgecolor="black")
for i, (v100, v150, dec) in enumerate(zip(vals_100, vals_150, decays)):
    ax.text(i - w/2, v100 + 0.005, f"{v100:.3f}", ha="center", fontsize=10)
    ax.text(i + w/2, v150 + 0.005, f"{v150:.3f}", ha="center", fontsize=10)
    color = "green" if dec >= 0 else "red"
    ax.text(i, max(v100, v150) + 0.02, f"{dec:+.1f}%", ha="center",
            fontsize=12, fontweight="bold", color=color)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Changed Acc", fontsize=12)
ax.set_title("OOD 外推衰减对比（t=100 → t=150）", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out3 = root / "wsl_ood_decay_bar.png"
fig.savefig(out3, dpi=140)
print(f"已绘图 {out3}")
plt.close(fig)

# === 图4：正交性热力图（three_chain）===
if "three_chain" in test_data and "orthogonality" in test_data["three_chain"]:
    orth = test_data["three_chain"]["orthogonality"]
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
    ax.set_title("WSL 真实 Mamba 三轴正交性", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out4 = root / "wsl_orthogonality_heatmap.png"
    fig.savefig(out4, dpi=140)
    print(f"已绘图 {out4}")
    plt.close(fig)

# === 图5：GRU vs Mamba 对比（changed_acc）===
gru_data = {}
gru_root = Path("results")
for m in models:
    gp = gru_root / f"run_{m}_seed0" / "metrics.json"
    if gp.exists():
        gru_data[m] = json.loads(gp.read_text(encoding="utf-8"))

if gru_data:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_labels = []
    gru_vals = []
    mamba_vals = []
    for m in models:
        if m in gru_data and m in test_data:
            x_labels.append(labels[m])
            gru_vals.append(gru_data[m].get("changed_acc", 0))
            mamba_vals.append(test_data[m].get("changed_acc", 0))
    x = np.arange(len(x_labels))
    w = 0.35
    ax.bar(x - w/2, gru_vals, w, label="GRU fallback (Windows)", color="#888888", edgecolor="black")
    ax.bar(x + w/2, mamba_vals, w, label="真实 Mamba (WSL)", color="#a570ff", edgecolor="black")
    for i, (gv, mv) in enumerate(zip(gru_vals, mamba_vals)):
        ax.text(i - w/2, gv + 0.005, f"{gv:.3f}", ha="center", fontsize=10)
        ax.text(i + w/2, mv + 0.005, f"{mv:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Changed Acc (test)", fontsize=12)
    ax.set_title("GRU fallback vs 真实 Mamba 对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out5 = root / "gru_vs_mamba.png"
    fig.savefig(out5, dpi=140)
    print(f"已绘图 {out5}")
    plt.close(fig)

print("\n=== 所有图绘制完成 ===")
print("\n=== test 集结果汇总 ===")
print(f"{'model':<22} {'acc_final':>10} {'changed_acc':>12}")
print("-" * 46)
for m in models:
    if m in test_data:
        d = test_data[m]
        print(f"{m:<22} {d['acc_final']:>10.4f} {d.get('changed_acc', 0):>12.4f}")

print("\n=== OOD 结果汇总 ===")
print(f"{'model':<22} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
print("-" * 48)
for m in models:
    if m in ood_data:
        c = ood_data[m]["changed_acc_curve"]
        a100 = c.get("100", 0)
        a150 = c.get("150", 0)
        dec = (a150 - a100) / a100 * 100 if a100 > 0 else 0
        print(f"{m:<22} {a100:>8.4f} {a150:>8.4f} {dec:>+7.1f}%")
