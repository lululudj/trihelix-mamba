# -*- coding: utf-8 -*-
"""
ThreeChainMamba2 SDD 真实多智能体数据对比图生成脚本
bookstore(7视频) vs nexus(12视频) —— 宇宙深色风
生成 4 张图到 e:\沐曦基金申请文件\figures\
"""
import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager

# ---------- 字体设置 ----------
for fname in ["SimHei", "Microsoft YaHei"]:
    if any(fname.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 宇宙深色风配色 ----------
BG_FIG = "#0a0a1a"      # 图背景 深紫黑
BG_AX = "#0d0d2b"       # 轴背景
GRID = "#2a2a4a"        # 网格
TXT = "#e0e0f0"         # 文字
C_BLUE = "#4a9eff"      # 蓝 (bookstore SSM)
C_PURPLE = "#b366ff"    # 紫 (nexus SSM)
C_PINK = "#ff6b9d"      # 粉
C_GOLD = "#ffd700"      # 金
C_CYAN = "#00d4ff"      # 青
C_GRAY = "#9a9ab0"      # 灰 (shuffle)
C_RED = "#ff5c5c"       # 红 (shuffle / baseline)

OUT_DIR = r"e:\沐曦基金申请文件\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 数据路径 ----------
P_BK_SSM = r"e:\three_chain_v3\results_stage2\sdd_30m_seed0_10k\ood_metrics_advanced.json"
P_BK_SHF = r"e:\three_chain_v3\results_stage2\sdd_shuffle_3k\ood_metrics_advanced.json"
P_NX_SSM = r"e:\three_chain_v3\results_stage2\nexus_30m_seed0_10k\ood_metrics_advanced.json"
P_NX_SHF = r"e:\three_chain_v3\results_stage2\nexus_shuffle_3k\ood_metrics_advanced.json"
P_BK_NEG = r"e:\three_chain_v3\results_stage2\sdd_30m_seed0_10k\negative_shadow.json"
P_NX_NEG = r"e:\three_chain_v3\results_stage2\nexus_30m_seed0_10k\negative_shadow.json"


# ---------- 工具函数 ----------
def load_json(path):
    if not os.path.isfile(path):
        print(f"[WARN] 文件不存在,跳过: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 读取失败 {path}: {e}")
        return None


def get_curve(data, key="position_iou"):
    """从 ood_metrics_advanced 结构提取 {t(int): value(float)} 曲线。
    兼容两种结构:顶层 curves 字典 / 直接 pos_iou_curve 字典。"""
    if data is None:
        return {}
    # 结构1: {"curves": {"1": {"position_iou": ...}, ...}}
    if isinstance(data.get("curves"), dict):
        out = {}
        for t, v in data["curves"].items():
            try:
                out[int(t)] = float(v.get(key, np.nan))
            except (ValueError, TypeError, AttributeError):
                continue
        return out
    # 结构2 (备用): {"pos_iou_curve": {"50": ...}}
    alias = {
        "position_iou": "pos_iou_curve",
        "changed_acc": "changed_acc_curve",
        "agent_id_acc": "agent_id_acc_curve",
    }
    fld = alias.get(key, f"{key}_curve")
    if isinstance(data.get(fld), dict):
        out = {}
        for t, v in data[fld].items():
            try:
                out[int(t)] = float(v)
            except (ValueError, TypeError):
                continue
        return out
    return {}


def window_avg(curve, center, half=5):
    """±half 邻域均值,避免单点抖动(尤其 t=100 训练边界深坑)。"""
    if not curve:
        return np.nan
    vals = []
    for t in range(center - half, center + half + 1):
        if t in curve:
            v = curve[t]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
    if not vals:
        return np.nan
    return float(np.mean(vals))


def smooth_curve(curve, half=5):
    """对整条曲线做中心滑动均值(窗口=2*half+1),返回排序后的 (xs, ys)。"""
    if not curve:
        return np.array([]), np.array([])
    ts = sorted(curve.keys())
    xs, ys = [], []
    for t in ts:
        v = window_avg(curve, t, half=half)
        if not np.isnan(v):
            xs.append(t)
            ys.append(v)
    return np.array(xs), np.array(ys)


def style_ax(ax):
    """应用深色风样式。"""
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TXT, labelcolor=TXT)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def glow_title(ax, title, subtitle=None):
    """带 glow 效果的标题。"""
    t = ax.set_title(title, color=TXT, fontsize=15, fontweight="bold", pad=16)
    t.set_path_effects([
        pe.withStroke(linewidth=3, foreground=C_CYAN, alpha=0.35),
        pe.withStroke(linewidth=6, foreground=C_PURPLE, alpha=0.15),
    ])
    if subtitle:
        ax.text(0.5, 1.0, subtitle, transform=ax.transAxes, color=TXT,
                fontsize=9.5, alpha=0.75, ha="center", va="bottom")


def style_legend(ax, loc="best"):
    leg = ax.legend(loc=loc, framealpha=0.55, facecolor="#1a1a3a",
                    edgecolor=GRID, labelcolor=TXT, fontsize=9)
    if leg:
        for txt in leg.get_texts():
            txt.set_color(TXT)
    return leg


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] 已保存: {path}")
    return path


# ---------- 加载数据 ----------
print("=== 加载数据 ===")
bk_ssm = load_json(P_BK_SSM)
bk_shf = load_json(P_BK_SHF)
nx_ssm = load_json(P_NX_SSM)
nx_shf = load_json(P_NX_SHF)
bk_neg = load_json(P_BK_NEG)
nx_neg = load_json(P_NX_NEG)

# 随机基线 (1/cell_types)
rand_base = 1.0 / 16.0  # 0.0625; 文档中给出 1/15=0.0667, 以数据文件为准
for d in (bk_ssm, nx_ssm, bk_shf, nx_shf):
    if d and "random_baseline_agent" in d:
        rand_base = float(d["random_baseline_agent"])
        break
print(f"随机基线 agent_id = {rand_base:.4f}")

# 提取曲线
bk_ssm_iou = get_curve(bk_ssm, "position_iou")
nx_ssm_iou = get_curve(nx_ssm, "position_iou")
bk_shf_iou = get_curve(bk_shf, "position_iou")
nx_shf_iou = get_curve(nx_shf, "position_iou")
bk_ssm_aid = get_curve(bk_ssm, "agent_id_acc")
nx_ssm_aid = get_curve(nx_ssm, "agent_id_acc")


# =====================================================================
# 图1: position_iou OOD 衰减对比
# =====================================================================
print("\n=== 图1: pos_iou 衰减对比 ===")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_FIG)
style_ax(ax)

lines_cfg = [
    ("bookstore SSM", bk_ssm_iou, C_BLUE, "-", 2.2),
    ("nexus SSM",    nx_ssm_iou, C_PURPLE, "-", 2.2),
    ("bookstore shuffle", bk_shf_iou, C_GRAY, "--", 1.8),
    ("nexus shuffle",     nx_shf_iou, C_GRAY, ":", 1.8),
]
for label, curve, color, ls, lw in lines_cfg:
    if not curve:
        print(f"[WARN] 跳过空曲线: {label}")
        continue
    xs, ys = smooth_curve(curve, half=5)
    if len(xs) == 0:
        print(f"[WARN] 曲线为空(平滑后): {label}")
        continue
    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.95)
    # 端点标记
    ax.scatter([xs[0]], [ys[0]], color=color, s=22, zorder=5, edgecolor=BG_AX, linewidth=0.5)

# 训练边界 & OOD 竖线
ax.axvline(100, color=C_GOLD, linestyle="--", linewidth=1.3, alpha=0.8)
ax.axvline(150, color=C_PINK, linestyle="--", linewidth=1.3, alpha=0.8)
ymax = ax.get_ylim()[1]
ax.text(100, ymax * 0.97, " t=100\n 训练边界", color=C_GOLD, fontsize=8.5, va="top", ha="left")
ax.text(150, ymax * 0.97, " t=150\n OOD", color=C_PINK, fontsize=8.5, va="top", ha="left")

ax.set_xlim(48, 152)
ax.set_xlabel("时间步 t", fontsize=11)
ax.set_ylabel("position_iou (窗口±5均值)", fontsize=11)
glow_title(ax, "多智能体长程外推 position_iou 衰减对比 (SDD 真实数据)",
           "bookstore(7视频) vs nexus(12视频) — t=100后为OOD外推区")
style_legend(ax, loc="upper right")

# 关键数值摘要
def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "N/A"
print("  关键点 pos_iou (window_avg ±5):")
for name, c in [("bookstore SSM", bk_ssm_iou), ("nexus SSM", nx_ssm_iou),
                ("bookstore shuffle", bk_shf_iou), ("nexus shuffle", nx_shf_iou)]:
    if c:
        print(f"    {name:20s} t50={fmt(window_avg(c,50))}  t100={fmt(window_avg(c,100))}  t150={fmt(window_avg(c,150))}")

save_fig(fig, "fig_pos_iou_decay_compare.png")


# =====================================================================
# 图2: 三链负面分身对比 (机制图)
# =====================================================================
print("\n=== 图2: 三链负面分身对比 ===")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_FIG)
style_ax(ax)

modes = ["normal", "ablate_s", "ablate_t", "ablate_c", "ablate_all"]
mode_labels = ["normal", "ablate_s", "ablate_t", "ablate_c", "ablate_all"]


def neg_pos_iou_at100(neg_data, mode):
    """从 negative_shadow 取某 mode 的 position_iou@100 (window_avg±5)。"""
    if not neg_data:
        return np.nan
    node = neg_data.get(mode, {}) if isinstance(neg_data, dict) else {}
    if not isinstance(node, dict):
        return np.nan
    # 兼容: node 含 curves 字典 / node 直接含 pos_iou@100
    if isinstance(node.get("curves"), dict):
        c = {}
        for t, v in node["curves"].items():
            try:
                c[int(t)] = float(v.get("position_iou", np.nan))
            except (ValueError, TypeError, AttributeError):
                continue
        return window_avg(c, 100, half=5)
    # 备用: 直接字段 pos_iou
    if "pos_iou" in node:
        try:
            return float(node["pos_iou"])
        except (ValueError, TypeError):
            return np.nan
    return np.nan


bk_vals = [neg_pos_iou_at100(bk_neg, m) for m in modes]
nx_vals = [neg_pos_iou_at100(nx_neg, m) for m in modes]
print("  bookstore 负面分身 pos_iou@100:", {m: fmt(v) for m, v in zip(modes, bk_vals)})
print("  nexus     负面分身 pos_iou@100:", {m: fmt(v) for m, v in zip(modes, nx_vals)})

x = np.arange(len(modes))
w = 0.36
bars_bk = ax.bar(x - w / 2, [v if not np.isnan(v) else 0 for v in bk_vals], w,
                 color=C_BLUE, label="bookstore (7视频)", edgecolor=BG_AX, linewidth=0.6, alpha=0.92)
bars_nx = ax.bar(x + w / 2, [v if not np.isnan(v) else 0 for v in nx_vals], w,
                 color=C_PURPLE, label="nexus (12视频)", edgecolor=BG_AX, linewidth=0.6, alpha=0.92)

# 数值标签
for bars, vals in [(bars_bk, bk_vals), (bars_nx, nx_vals)]:
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, fmt(v),
                    ha="center", va="bottom", color=TXT, fontsize=8.5)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.005, "N/A",
                    ha="center", va="bottom", color=C_RED, fontsize=8)

# 标注 ablate_all 相对 normal 下降百分比
def drop_pct(vals):
    n, a = vals[0], vals[-1]
    if np.isnan(n) or np.isnan(a) or n == 0:
        return None
    return (a - n) / n * 100.0

bk_drop = drop_pct(bk_vals)
nx_drop = drop_pct(nx_vals)
ann_y = max([v for v in (bk_vals + nx_vals) if not np.isnan(v)] + [0.3]) * 1.02
if bk_drop is not None:
    ax.annotate(f"bookstore\nablate_all ↓{bk_drop:+.1f}%", xy=(4 - w / 2, bk_vals[-1]),
                xytext=(4 - w / 2, ann_y), color=C_CYAN, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_CYAN, lw=1.0))
if nx_drop is not None:
    ax.annotate(f"nexus\nablate_all ↓{nx_drop:+.1f}%", xy=(4 + w / 2, nx_vals[-1]),
                xytext=(4 + w / 2, ann_y * 0.86), color=C_GOLD, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0))

ax.set_xticks(x)
ax.set_xticklabels(mode_labels, color=TXT, fontsize=10)
ax.set_ylabel("pos_iou @100 (window_avg±5)", fontsize=11)
ax.set_xlabel("消融模式", fontsize=11)
glow_title(ax, "三链 SSM 演化贡献对比 (负面分身消融)",
           "ablate_all 后 pos_iou 大降 → 三链有实质贡献")
style_legend(ax, loc="upper right")
ax.set_ylim(0, max([v for v in (bk_vals + nx_vals) if not np.isnan(v)] + [0.3]) * 1.25)

save_fig(fig, "fig_negative_shadow_compare.png")


# =====================================================================
# 图3: shuffle sanity 验证
# =====================================================================
print("\n=== 图3: shuffle sanity 验证 ===")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_FIG)
style_ax(ax)

bk_ssm_v = window_avg(bk_ssm_iou, 100) if bk_ssm_iou else np.nan
nx_ssm_v = window_avg(nx_ssm_iou, 100) if nx_ssm_iou else np.nan
bk_shf_v = window_avg(bk_shf_iou, 100) if bk_shf_iou else np.nan
nx_shf_v = window_avg(nx_shf_iou, 100) if nx_shf_iou else np.nan

scenes = ["bookstore\n(7视频)", "nexus\n(12视频)"]
ssm_vals = [bk_ssm_v, nx_ssm_v]
shf_vals = [bk_shf_v, nx_shf_v]

x = np.arange(len(scenes))
w = 0.36
b1 = ax.bar(x - w / 2, [v if not np.isnan(v) else 0 for v in ssm_vals], w,
            color=C_BLUE, label="SSM (真实训练)", edgecolor=BG_AX, linewidth=0.6, alpha=0.92)
b2 = ax.bar(x + w / 2, [v if not np.isnan(v) else 0 for v in shf_vals], w,
            color=C_RED, label="shuffle (标签打乱)", edgecolor=BG_AX, linewidth=0.6, alpha=0.92)

for bars, vals in [(b1, ssm_vals), (b2, shf_vals)]:
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, fmt(v),
                    ha="center", va="bottom", color=TXT, fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.004, "N/A",
                    ha="center", va="bottom", color=C_RED, fontsize=8)

# 标注倍数
for i, (s, h) in enumerate(zip(ssm_vals, shf_vals)):
    if not np.isnan(s) and not np.isnan(h) and h > 1e-9:
        ratio = s / h
        ax.text(i, max(s, h) + 0.025, f"×{ratio:.1f}", ha="center", va="bottom",
                color=C_GOLD, fontsize=10.5, fontweight="bold")
    elif not np.isnan(s) and (np.isnan(h) or h <= 1e-9):
        ax.text(i, s + 0.025, "shuffle≈0 → SSM有效", ha="center", va="bottom",
                color=C_GOLD, fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(scenes, color=TXT, fontsize=10)
ax.set_ylabel("pos_iou @100 (window_avg±5)", fontsize=11)
glow_title(ax, "shuffle_labels sanity 验证 — position_iou 指标有效性",
           "两个场景 shuffle pos_iou≈0, SSM 显著高 → position_iou 是有效指标")
style_legend(ax, loc="upper right")
ymax3 = max([v for v in (ssm_vals + shf_vals) if not np.isnan(v)] + [0.3])
ax.set_ylim(0, ymax3 * 1.30)

print(f"  bookstore: SSM={fmt(bk_ssm_v)} shuffle={fmt(bk_shf_v)} ratio={fmt(bk_ssm_v/bk_shf_v) if bk_shf_v>1e-9 else 'inf'}")
print(f"  nexus:     SSM={fmt(nx_ssm_v)} shuffle={fmt(nx_shf_v)} ratio={fmt(nx_ssm_v/nx_shf_v) if nx_shf_v>1e-9 else 'inf'}")

save_fig(fig, "fig_shuffle_sanity.png")


# =====================================================================
# 图4: agent 身份学习能力对比
# =====================================================================
print("\n=== 图4: agent_id 学习能力对比 ===")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_FIG)
style_ax(ax)

bk_aid = window_avg(bk_ssm_aid, 100) if bk_ssm_aid else np.nan
nx_aid = window_avg(nx_ssm_aid, 100) if nx_ssm_aid else np.nan
print(f"  bookstore agent_id@100 = {fmt(bk_aid)}  (随机 {rand_base:.4f})")
print(f"  nexus     agent_id@100 = {fmt(nx_aid)}  (随机 {rand_base:.4f})")

scenes = ["bookstore\n(7视频)", "nexus\n(12视频)"]
vals = [bk_aid, nx_aid]
colors = [C_BLUE, C_PURPLE]

# 随机基线水平线
ax.axhline(rand_base, color=C_RED, linestyle="--", linewidth=1.6, alpha=0.85,
           label=f"随机基线 1/15 = {rand_base:.4f}")

x = np.arange(len(scenes))
bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in vals], 0.5,
              color=colors, edgecolor=BG_AX, linewidth=0.6, alpha=0.92,
              label="SSM agent_id@100")

for b, v in zip(bars, vals):
    if not np.isnan(v):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, fmt(v),
                ha="center", va="bottom", color=TXT, fontsize=10, fontweight="bold")

# 标注是否超过随机
for i, v in enumerate(vals):
    if np.isnan(v):
        continue
    if v > rand_base:
        ax.text(i, v + 0.012, "学到 ↑", ha="center", va="bottom", color=C_GOLD,
                fontsize=9.5, fontweight="bold")
    else:
        ax.text(i, v + 0.012, "未学到 ↓", ha="center", va="bottom", color=C_RED,
                fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(scenes, color=TXT, fontsize=10)
ax.set_ylabel("agent_id_acc @100 (window_avg±5)", fontsize=11)
glow_title(ax, "多场景数据让 SSM 学到 agent 身份 (agent_id_acc > 随机基线)",
           f"nexus={fmt(nx_aid)} > 随机{rand_base:.3f}(学到) | bookstore={fmt(bk_aid)} < 随机(未学到)")
style_legend(ax, loc="upper right")
ymax4 = max([v for v in vals if not np.isnan(v)] + [rand_base]) * 1.45
ax.set_ylim(0, ymax4)

save_fig(fig, "fig_agent_id_learning.png")

print("\n=== 全部完成 ===")
print(f"输出目录: {OUT_DIR}")
