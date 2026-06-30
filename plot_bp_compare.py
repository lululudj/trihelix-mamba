"""绘制 ThreeChain vs ThreeChainBP v1/v2 对比图（test + OOD）。

v1：tanh 门控（初始 0）→ 门控死锁，BP 层空操作（无效实验）
v2：移除门控，标准 transformer 残差 → 强制 BP 层生效（真正测试）
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("results_wsl")


def load(model_dir, kind):
    """kind: 'metrics' or 'ood_metrics'"""
    p = root / model_dir / f"{kind}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# 对比配置：three_chain (基线) vs BP v1 (无效) vs BP v2 (真正测试)
configs = [
    ("run_three_chain_seed0", "Three-Chain", "#a570ff", "-", "o"),
    ("run_three_chain_bp_seed0", "Three-Chain + BP v1（门控死锁）", "#999999", ":", "x"),
    ("run_three_chain_bp_v2_seed0", "Three-Chain + BP v2（无门控）", "#ff4d6d", "--", "s"),
]

# === 图1：test 集 changed_acc 对比 ===
fig, ax = plt.subplots(figsize=(10, 5.5))
for model_dir, label, color, ls, marker in configs:
    d = load(model_dir, "metrics")
    if d is None:
        print(f"skip {model_dir} (no metrics)")
        continue
    curve = d["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ys, ls, color=color, linewidth=2.5, marker=marker,
            markersize=7, label=label)
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t", fontsize=12)
ax.set_title("test 集 changed_acc 对比（T=100）：碱基对效应", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 30, 60, 100])
ax.set_xticklabels(["1", "5", "10", "30", "60", "100"])
fig.tight_layout()
out1 = root / "bp_test_compare.png"
fig.savefig(out1, dpi=140)
print(f"已绘图 {out1}")
plt.close(fig)

# === 图2：OOD T=150 changed_acc 对比 ===
fig, ax = plt.subplots(figsize=(11, 5.5))
for model_dir, label, color, ls, marker in configs:
    d = load(model_dir, "ood_metrics")
    if d is None:
        print(f"skip {model_dir} (no ood)")
        continue
    curve = d["changed_acc_curve"]
    xs = sorted(int(k) for k in curve.keys())
    ys = [curve[str(x)] for x in xs]
    ax.plot(xs, ls, color=color, linewidth=2.5, marker=marker,
            markersize=6, label=label)
    if "100" in curve:
        ax.scatter([100], [curve["100"]], color=color, s=100, zorder=5, edgecolors="black")
    if "150" in curve:
        ax.scatter([150], [curve["150"]], color=color, s=100, zorder=5, edgecolors="black")
ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7, label="训练边界 t=100")
ax.axvspan(100, 150, alpha=0.1, color="red", label="OOD 外推区")
ax.set_xlabel("步数 t", fontsize=12)
ax.set_ylabel("Changed Acc@t", fontsize=12)
ax.set_title("OOD 长程外推对比（T=150）：碱基对效应", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 155)
fig.tight_layout()
out2 = root / "bp_ood_compare.png"
fig.savefig(out2, dpi=140)
print(f"已绘图 {out2}")
plt.close(fig)

# === 图3：OOD 衰减柱状图对比 ===
fig, ax = plt.subplots(figsize=(9, 5.5))
labels_bar = []
vals_100 = []
vals_150 = []
decays = []
colors_bar = []
for model_dir, label, color, ls, marker in configs:
    d = load(model_dir, "ood_metrics")
    if d is None:
        continue
    curve = d["changed_acc_curve"]
    if "100" in curve and "150" in curve:
        labels_bar.append(label.replace("Three-Chain + ", "").replace("Three-Chain", "基线 Three-Chain"))
        vals_100.append(curve["100"])
        vals_150.append(curve["150"])
        decays.append((curve["150"] - curve["100"]) / curve["100"] * 100 if curve["100"] > 0 else 0)
        colors_bar.append(color)
x = np.arange(len(labels_bar))
w = 0.35
ax.bar(x - w/2, vals_100, w, label="t=100 (训练长度)", color="#5ac8fa", edgecolor="black")
ax.bar(x + w/2, vals_150, w, label="t=150 (OOD 外推)", color="#ff7043", edgecolor="black")
for i, (v100, v150, dec) in enumerate(zip(vals_100, vals_150, decays)):
    ax.text(i - w/2, v100 + 0.005, f"{v100:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, v150 + 0.005, f"{v150:.3f}", ha="center", fontsize=9)
    color = "green" if dec >= 0 else "red"
    ax.text(i, max(v100, v150) + 0.018, f"{dec:+.1f}%", ha="center",
            fontsize=11, fontweight="bold", color=color)
ax.set_xticks(x)
ax.set_xticklabels(labels_bar, fontsize=9)
ax.set_ylabel("Changed Acc", fontsize=12)
ax.set_title("碱基对效应：OOD 外推衰减对比（t=100 → t=150）", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out3 = root / "bp_ood_decay_bar.png"
fig.savefig(out3, dpi=140)
print(f"已绘图 {out3}")
plt.close(fig)

# === 汇总打印 ===
print("\n=== test 集对比 ===")
print(f"{'model':<40} {'acc_final':>10} {'changed_acc':>12}")
print("-" * 64)
for model_dir, label, _, _, _ in configs:
    d = load(model_dir, "metrics")
    if d:
        print(f"{label:<40} {d['acc_final']:>10.4f} {d.get('changed_acc', 0):>12.4f}")

print("\n=== OOD 对比 ===")
print(f"{'model':<40} {'ch@100':>8} {'ch@150':>8} {'decay':>8}")
print("-" * 66)
for model_dir, label, _, _, _ in configs:
    d = load(model_dir, "ood_metrics")
    if d:
        c = d["changed_acc_curve"]
        a100 = c.get("100", 0)
        a150 = c.get("150", 0)
        dec = (a150 - a100) / a100 * 100 if a100 > 0 else 0
        print(f"{label:<40} {a100:>8.4f} {a150:>8.4f} {dec:>+7.1f}%")

# 正交性对比（如果有）
print("\n=== 正交性对比 ===")
print(f"{'model':<40} {'h_s_h_c':>10} {'h_s_h_t':>10} {'h_c_h_t':>10}")
print("-" * 72)
for model_dir, label, _, _, _ in configs:
    d = load(model_dir, "metrics")
    if d and "orthogonality" in d:
        o = d["orthogonality"]
        print(f"{label:<40} {o.get('h_s_h_c', 0):>+10.4f} {o.get('h_s_h_t', 0):>+10.4f} {o.get('h_c_h_t', 0):>+10.4f}")
