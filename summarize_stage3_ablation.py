"""阶段 3.2 消融实验结果汇总脚本

读 baseline + 4 个 ablation 的 ood_metrics.json, 画消融对比图 + 写判定报告。

输入:
    results_cloud/run_30m_seed2_500/ood_metrics.json   (baseline, 30M seed2 @500步)
    results_stage3/ablation/no_spatial/ood_metrics.json
    results_stage3/ablation/no_temporal/ood_metrics.json
    results_stage3/ablation/no_causal/ood_metrics.json
    results_stage3/ablation/no_all/ood_metrics.json

输出:
    paper/figures/stage3_ablation.png   (5 个 decay 对比 + changed_acc 曲线)
    paper/stage3_ablation.md           (消融实验报告 + 判定结论)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    if any(f.lower() in fn.name.lower() for fn in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).parent.resolve()
BASELINE_DIR = BASE / "results_cloud" / "run_30m_seed2_500"
ABLATION_DIR = BASE / "results_stage3" / "ablation"
FIG_DIR = BASE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = BASE / "paper" / "stage3_ablation.md"

# 宇宙深色风配色
BG = "#0a0e27"
PANEL = "#1a1f3a"
GRID = "#2a3050"
TEXT = "#e0e6ff"
COLORS = {
    "baseline":    "#ffd700",  # 金
    "no_spatial":  "#5b8cff",  # 蓝
    "no_temporal": "#a866ff",  # 紫
    "no_causal":   "#ff5b8c",  # 粉
    "no_all":      "#ff7847",  # 橙红
}
LABELS = {
    "baseline":    "Baseline (三链全开)",
    "no_spatial":  "消融空间链",
    "no_temporal": "消融时间链",
    "no_causal":   "消融因果链",
    "no_all":      "三链全消融",
}


def load_ood(path):
    """读 ood_metrics.json, 不存在返回 None."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def window_decay(curve, t_ref=100, t_final=150, win=5):
    """窗口平滑 decay (±win 邻域均值), 与 summarize_all_experiments.py 一致."""
    def avg_at(t):
        vals = []
        for dt in range(-win, win + 1):
            k = str(t + dt)
            if k in curve:
                vals.append(curve[k])
        return sum(vals) / len(vals) if vals else None
    a_ref = avg_at(t_ref)
    a_final = avg_at(t_final)
    if a_ref is None or a_final is None or a_ref == 0:
        return None
    return (a_final - a_ref) / a_ref * 100


def main():
    print("=== 阶段 3.2 消融实验汇总 ===")
    # 收集所有结果
    runs = {}
    runs["baseline"] = load_ood(BASELINE_DIR / "ood_metrics.json")
    for name in ["no_spatial", "no_temporal", "no_causal", "no_all"]:
        runs[name] = load_ood(ABLATION_DIR / name / "ood_metrics.json")

    # 汇报加载状态
    print("\n--- 结果加载状态 ---")
    for k, v in runs.items():
        if v is None:
            print(f"  {k}: [未找到]")
        else:
            decay = v.get("ood_decay_pct", 0) * 100
            ch = v.get("changed_acc", 0)
            print(f"  {k}: decay={decay:+.2f}%, changed_acc@150={ch:.4f}")

    # 检查至少有 baseline + 1 个 ablation
    available = [k for k, v in runs.items() if v is not None]
    if "baseline" not in available:
        print("\n[ERROR] baseline 结果缺失, 无法对比")
        return
    if len(available) < 2:
        print("\n[ERROR] 至少需要 baseline + 1 个 ablation 结果")
        return

    # ===== 画图 =====
    print("\n--- 生成图表 ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

    # --- Fig A: decay bar chart ---
    ax = axes[0]
    ax.set_facecolor(PANEL)
    names = list(runs.keys())
    decays = [runs[n].get("ood_decay_pct", 0) * 100 if runs[n] else np.nan for n in names]
    colors = [COLORS[n] for n in names]
    labels = [LABELS[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, decays, color=colors, alpha=0.85, edgecolor=colors, linewidth=2)
    # ±2% 阈值线
    ax.axhspan(-2, 2, alpha=0.15, color="#00ff88", label="±2% 不退化区间")
    ax.axhline(0, color=TEXT, linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("OOD decay (t=100→150) [%]", color=TEXT)
    ax.set_title("消融对 OOD 长程外推衰减的影响", color=TEXT, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.2, color=GRID)
    # 数值标注
    for bar, d in zip(bars, decays):
        if not np.isnan(d):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{d:+.2f}%", ha="center",
                    va="bottom" if d >= 0 else "top",
                    color=TEXT, fontsize=9, fontweight="bold")
    legend = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # --- Fig B: changed_acc 曲线 ---
    ax = axes[1]
    ax.set_facecolor(PANEL)
    for name in names:
        if runs[name] is None:
            continue
        curve = runs[name].get("changed_acc_curve", {})
        if not curve:
            continue
        ts = sorted(int(k) for k in curve.keys())
        vals = [curve[str(t)] for t in ts]
        ax.plot(ts, vals, color=COLORS[name], label=LABELS[name],
                linewidth=2, alpha=0.9)
    # 训练上限标记
    ax.axvline(100, color="#ffaa55", linestyle="--", alpha=0.6, label="训练上限 t=100")
    ax.set_xlabel("时间步 t", color=TEXT)
    ax.set_ylabel("changed_acc", color=TEXT)
    ax.set_title("各消融模型 changed_acc 随 t 演化", color=TEXT, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.2, color=GRID)
    legend = ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "stage3_ablation.png"
    plt.savefig(fig_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  [OK] 图表: {fig_path}")

    # ===== 写报告 =====
    print("\n--- 生成报告 ---")
    # 计算窗口 decay
    win_decays = {}
    for name in names:
        if runs[name] is None:
            win_decays[name] = None
            continue
        curve = runs[name].get("changed_acc_curve", {})
        win_decays[name] = window_decay(curve)

    # 判定
    b_decay = decays[0]  # baseline
    conclusions = []
    for name in ["no_spatial", "no_temporal", "no_causal"]:
        if runs[name] is None:
            conclusions.append(f"- **{LABELS[name]}**: 未完成, 无法判定")
            continue
        d = decays[names.index(name)]
        if abs(d) > 2:
            tag = "贡献不退化"
            conclusions.append(f"- **{LABELS[name]}**: decay={d:+.2f}% (|decay|>2%) → 该链**{tag}**")
        else:
            tag = "非主因"
            conclusions.append(f"- **{LABELS[name]}**: decay={d:+.2f}% (±2%内) → 该链**{tag}**")

    # 全消融判定
    if runs["no_all"] is not None:
        d_all = decays[names.index("no_all")]
        if d_all < -6:
            conclusions.append(f"- **{LABELS['no_all']}**: decay={d_all:+.2f}% (<-6%) → 三链**协同贡献**不退化")
        elif abs(d_all) < 2:
            conclusions.append(f"- **{LABELS['no_all']}**: decay={d_all:+.2f}% (±2%内) → 不退化来自**残差融合机制本身**")
        else:
            conclusions.append(f"- **{LABELS['no_all']}**: decay={d_all:+.2f}% → 介于协同与残差之间, 需进一步分析")

    report = f"""# 阶段 3.2 消融实验报告

> 生成时间: 自动生成
> 基线: 30M seed2 @500步 (decay={b_decay:+.2f}%, 三链全开)
> 消融策略: 置零某链输出 (保持参数量 26.59M 不变, 严格控制变量)

## 实验设置

- **基线**: ThreeChainMamba2 30M (d_model=768, n_layers=2, seed=2, 500步)
- **消融方式**: 在残差融合前 `x = norm_fuse(x + x_s + x_t + c_inject)` 置零对应链输出
- **参数量**: 所有模型相同 26.59M (置零输出不删模块)
- **评估**: OOD T=150 (训练 T=100), 报告 changed_acc 的 t=100→150 decay

## 结果汇总

| 模型 | ablate_s | ablate_t | ablate_c | decay% | 窗口decay% | changed_acc@150 |
|------|----------|----------|----------|--------|-----------|----------------|
"""
    for name in names:
        if runs[name] is None:
            report += f"| {LABELS[name]} | - | - | - | [未完成] | - | - |\n"
            continue
        abl = {"no_spatial": (True, False, False),
               "no_temporal": (False, True, False),
               "no_causal": (False, False, True),
               "no_all": (True, True, True),
               "baseline": (False, False, False)}[name]
        d = decays[names.index(name)]
        wd = win_decays[name]
        ch = runs[name].get("changed_acc", 0)
        wd_str = f"{wd:+.2f}%" if wd is not None else "-"
        report += f"| {LABELS[name]} | {abl[0]} | {abl[1]} | {abl[2]} | {d:+.2f}% | {wd_str} | {ch:.4f} |\n"

    report += f"""
## 判定结论

{chr(10).join(conclusions)}

## 与 3.1 发现的对照

3.1 探针发现:
- H1 支持: 时间链 h_t 在 OOD 区稳定
- H2 推翻: 空间链 h_s 未显著衰减
- H3 支持: 因果链 h_c 极其稳定
- 核心结论: 不退化来自**架构整体稳定性** (三链并行 + 残差融合), 而非单链独撑

3.2 消融预期:
- 若单链消融后 decay 仍 ±2% → 印证"非单链独撑"
- 关注全消融 no_all: 区分"融合机制贡献" vs "单链冗余"

## 判定标准

- 消融某单链后 |decay| > 2% → 该链**贡献**不退化
- 消融某单链后 decay 仍 ±2% → 该链**非主因**
- no_all decay <-6% → 三链**协同**贡献
- no_all decay 仍 ±2% → 不退化来自**残差融合机制本身**

## 产出物

- 图表: `paper/figures/stage3_ablation.png`
- 本报告: `paper/stage3_ablation.md`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  [OK] 报告: {REPORT_PATH}")

    print("\n=== 完成 ===")
    print(f"图表: {fig_path}")
    print(f"报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
