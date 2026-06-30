"""Regenerate all figures with ASCII-safe text (no garbled glyphs) + add fig7 ablation.

Fixes: replace Unicode chars (-> +/- ~ *) that may render as boxes on some fonts.
Adds: fig7_ablation.png — BPv1 / BPv2 / Transformer-tiny vs baseline.
Does NOT touch PROFESSIONAL_REPORT.md (manual edits preserved).
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.resolve()
OLD_BASE = HERE / "results_wsl" / "benchmark_multiseed"
NEW_BASE = HERE / "results_wsl" / "benchmark_mamba2"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]

MODELS = [
    ("ThreeChain (old)",     "three_chain",        OLD_BASE, "#888899", False),
    ("SingleChain",          "single_chain",       OLD_BASE, "#4dd0e1", False),
    ("Transformer",          "transformer",        OLD_BASE, "#ba68c8", False),
    ("ThreeChainMamba2 (A)", "three_chain_mamba2", NEW_BASE, "#ffd54f", True),
]


def setup_style():
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor":  "#0a0e27",
        "axes.facecolor":    "#0a0e27",
        "savefig.facecolor": "#0a0e27",
        "axes.edgecolor":    "#3a4068",
        "axes.labelcolor":   "#e0e0ff",
        "axes.titlecolor":   "#ffffff",
        "xtick.color":       "#a0a0c0",
        "ytick.color":       "#a0a0c0",
        "grid.color":        "#1f2547",
        "grid.alpha":        0.6,
        "text.color":        "#e0e0ff",
        "legend.facecolor":  "#141838",
        "legend.edgecolor":  "#3a4068",
        "legend.labelcolor": "#e0e0ff",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.unicode_minus": False,
    })


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_log(base, model_dir, seed):
    f = base / f"{model_dir}_seed{seed}" / "log.jsonl"
    if not f.exists():
        return [], []
    train, val = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (val if r.get("event") == "val" else train).append(r)
    return train, val


def collect_all():
    data = {}
    for disp, prefix, base, color, is_star in MODELS:
        rows = []
        for seed in SEEDS:
            d = base / f"{prefix}_seed{seed}"
            rows.append({
                "seed": seed,
                "summary": load_json(d / "summary.json") or {},
                "ood": load_json(d / "ood_metrics.json") or {},
                "train_log": load_log(base, prefix, seed)[0],
            })
        data[disp] = {"rows": rows, "color": color, "is_star": is_star}
    return data


def curve_mean_std(curves, max_t=150):
    ts = list(range(1, max_t + 1))
    arr = []
    for c in curves:
        if c:
            arr.append([float(c.get(str(t), np.nan)) for t in ts])
    if not arr:
        return ts, np.array([]), np.array([]), 0
    arr = np.array(arr)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return ts, mean, std, arr.shape[0]


def fig1_training(data):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax_loss, ax_ch = axes
    for disp, info in data.items():
        row = next((r for r in info["rows"] if r["seed"] == 42), None)
        if not row or not row["train_log"]:
            continue
        log = row["train_log"]
        steps = [r["step"] for r in log]
        loss = [r.get("loss", r.get("loss_final", 0)) for r in log]
        ch = [r.get("changed_acc", np.nan) for r in log]
        lw = 2.6 if info["is_star"] else 1.6
        ax_loss.plot(steps, loss, color=info["color"], lw=lw, alpha=0.9, label=disp)
        ax_ch.plot(steps, ch, color=info["color"], lw=lw, alpha=0.9, label=disp)
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Dynamics (seed=42)", fontweight="bold")
    ax_loss.legend(loc="upper right")
    ax_ch.set_ylabel("Training changed_acc")
    ax_ch.set_xlabel("Training Step")
    ax_ch.set_ylim(0, 0.75)
    ax_ch.axhline(0, color="#555577", lw=0.8, ls=":")
    ax_ch.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig1_training_dynamics.png")


def fig2_ood_curve(data):
    fig, ax = plt.subplots(figsize=(13, 7))
    for disp, info in data.items():
        curves = [r["ood"].get("changed_acc_curve", {}) for r in info["rows"]]
        ts, mean, std, n = curve_mean_std(curves, 150)
        if len(mean) == 0:
            continue
        lw = 3.0 if info["is_star"] else 1.8
        ax.plot(ts, mean, color=info["color"], lw=lw, label=f"{disp} (n={n})")
        ax.fill_between(ts, mean - std, mean + std, color=info["color"],
                        alpha=0.22 if info["is_star"] else 0.15, linewidth=0)
    ax.axvline(100, color="#ff6b9d", lw=1.8, ls="--", alpha=0.7)
    ax.axvspan(100, 150, color="#ff6b9d", alpha=0.06, zorder=0)
    ax.text(101, 0.02, "OOD zone\n(T=100 -> 150)", color="#ff8fb8", fontsize=9, va="bottom")
    ax.text(50, 0.02, "Training zone (T<=100)", color="#a0a0c0", fontsize=9, va="bottom", ha="center")
    ax.set_xlabel("Time Step t")
    ax.set_ylabel("changed_acc (cells that changed)")
    ax.set_title("OOD Long-range Extrapolation: changed_acc vs t  (5 seeds, mean+/-std)",
                 fontweight="bold")
    ax.set_xlim(1, 150)
    ax.set_ylim(0, 0.75)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_ood_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig2_ood_curve.png")


def fig3_ood_decay(data):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    names, means, stds, colors = [], [], [], []
    for disp, info in data.items():
        decays = [r["ood"].get("ood_decay_100_to_150") for r in info["rows"]
                  if r["ood"].get("ood_decay_100_to_150") is not None]
        if not decays:
            continue
        names.append(disp)
        means.append(np.mean(decays))
        stds.append(np.std(decays, ddof=1) if len(decays) > 1 else 0.0)
        colors.append(info["color"])
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6,
                  edgecolor="#e0e0ff", linewidth=1.2, alpha=0.88, width=0.6)
    for i, (disp, info) in enumerate(data.items()):
        if info["is_star"] and i < len(bars):
            bars[i].set_linewidth(2.4)
    ax.axhline(0, color="#ff6b9d", lw=1.5, ls="--", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("OOD Decay (ch_acc@t150 - ch_acc@t100)")
    ax.set_title("OOD Extrapolation Decay (5 seeds, mean+/-std)  --  closer to 0 = better",
                 fontweight="bold")
    for i, (m, s) in enumerate(zip(means, stds)):
        y = m - s - 0.0015 if m < 0 else m + s + 0.0005
        ax.text(i, y, f"{m:+.4f}", ha="center", va="top" if m < 0 else "bottom",
                fontsize=10, color="#e0e0ff", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ood_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig3_ood_decay.png")


def fig4_param_efficiency(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    for disp, info in data.items():
        xs, ys = [], []
        for r in info["rows"]:
            params = r["summary"].get("params", 0)
            ch = r["ood"].get("changed_acc", 0)
            if params > 0:
                xs.append(params / 1e6)
                ys.append(ch)
        if not xs:
            continue
        size = 180 if info["is_star"] else 110
        edge = "#ffffff" if info["is_star"] else info["color"]
        ax.scatter(xs, ys, s=size, c=info["color"], edgecolors=edge,
                   linewidths=2.2 if info["is_star"] else 1.0, alpha=0.88, label=disp, zorder=3)
        ax.scatter([np.mean(xs)], [np.mean(ys)], s=350, c=info["color"], marker="*",
                   edgecolors="#ffffff", linewidths=1.6, alpha=1.0, zorder=4)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("OOD changed_acc @ t=150")
    ax.set_title("Parameter Efficiency: Model Size vs OOD Performance  (* = 5-seed mean)",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_param_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig4_param_efficiency.png")


def fig5_collapse(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax1, ax2 = axes
    models_short = ["ThreeChain\n(old)", "ThreeChainMamba2\n(new A)"]
    zero_ratio = [97.7, 17.2]
    colors = ["#888899", "#ffd54f"]
    bars1 = ax1.bar(models_short, zero_ratio, color=colors, edgecolor="#e0e0ff",
                    linewidth=1.4, width=0.55)
    bars1[1].set_linewidth(2.4)
    ax1.axhline(90, color="#ff5252", lw=1.5, ls="--", alpha=0.8)
    ax1.text(0.5, 91, "collapse threshold 90%", color="#ff8a8a", fontsize=9, ha="center")
    ax1.set_ylabel("Training-domain zero_ratio (%)")
    ax1.set_title("Collapse Diagnosis: Prediction = 0 ratio", fontweight="bold")
    ax1.set_ylim(0, 110)
    for b, v in zip(bars1, zero_ratio):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
                 color="#e0e0ff", fontweight="bold", fontsize=12)
    ax1.text(0, 50, "[!] DEGENERATED\n(all-zero output)", ha="center",
             color="#ff8a8a", fontsize=9, fontweight="bold")
    ax1.text(1, 8, "[OK] ALIVE\n(real learning)", ha="center",
             color="#8aff8a", fontsize=9, fontweight="bold")
    names, stds, colors2 = [], [], []
    for disp, info in data.items():
        chs = [r["ood"].get("changed_acc", 0) for r in info["rows"]]
        std = np.std(chs, ddof=1) if len(chs) > 1 else 0
        names.append(disp.replace(" ", "\n", 1))
        stds.append(std)
        colors2.append(info["color"])
    x = np.arange(len(names))
    bars2 = ax2.bar(x, stds, color=colors2, edgecolor="#e0e0ff", linewidth=1.4, width=0.6)
    for i, (disp, info) in enumerate(data.items()):
        if info["is_star"] and i < len(bars2):
            bars2[i].set_linewidth(2.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("5-seed std of OOD changed_acc")
    ax2.set_title("Reality Check: Seed Variance  (std=0 -> BUG illusion)", fontweight="bold")
    for i, v in enumerate(stds):
        label = f"{v:.4f}\n[OK] real" if v > 0.0001 else f"{v:.4f}\n[!] BUG"
        ax2.text(i, v + 0.0003, label, ha="center",
                 color="#8aff8a" if v > 0.0001 else "#ff8a8a", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(stds) * 1.35 if stds else 0.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_collapse_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig5_collapse_diagnosis.png")


def fig6_seed_variance(data):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    names, box_data, colors = [], [], []
    for disp, info in data.items():
        chs = [r["ood"].get("changed_acc", 0) for r in info["rows"]]
        if chs:
            names.append(disp)
            box_data.append(chs)
            colors.append(info["color"])
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#ffffff", lw=2),
                    whiskerprops=dict(color="#a0a0c0"),
                    capprops=dict(color="#a0a0c0"),
                    flierprops=dict(marker="o", markerfacecolor="#ff6b9d", markersize=6, alpha=0.8))
    for patch, c, (disp, info) in zip(bp["boxes"], colors, data.items()):
        patch.set_facecolor(c)
        patch.set_alpha(0.75 if info["is_star"] else 0.55)
        patch.set_edgecolor("#ffffff" if info["is_star"] else c)
        patch.set_linewidth(2.2 if info["is_star"] else 1.0)
    for i, (chs, c) in enumerate(zip(box_data, colors)):
        xs = np.random.normal(i + 1, 0.04, size=len(chs))
        ax.scatter(xs, chs, c=c, s=45, edgecolors="#ffffff", linewidths=0.8, zorder=4, alpha=0.9)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("OOD changed_acc @ t=150")
    ax.set_title("Seed Variance: 5-seed OOD Performance Distribution\n"
                 "(flat box = std=0 -> BUG illusion; spread box = real learning)",
                 fontweight="bold")
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_seed_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig6_seed_variance.png")


def fig7_ablation():
    """NEW: Ablation study — BPv1 / BPv2 / Transformer-tiny vs baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Left: OOD changed_acc@150 bar chart
    labels = ["Baseline\n(Mamba2)", "BPv1\n(in-chain)", "BPv2\n(cross-chain)", "Transformer\n(same params)"]
    ch_acc = [0.5989, 0.5430, 0.5991, 0.0]
    colors = ["#ffd54f", "#ff5252", "#4dd0e1", "#ba68c8"]
    bars = ax1.bar(labels, ch_acc, color=colors, edgecolor="#e0e0ff", linewidth=1.4, width=0.6)
    bars[0].set_linewidth(2.4)
    ax1.axhline(0, color="#555577", lw=0.8)
    ax1.set_ylabel("OOD changed_acc @ t=150")
    ax1.set_title("Ablation: OOD Performance (2000 steps, seed=42)", fontweight="bold")
    ax1.set_ylim(0, 0.7)
    for b, v in zip(bars, ch_acc):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015,
                 f"{v:.4f}" if v > 0.01 else "0.0000\n(collapse)", ha="center",
                 color="#e0e0ff", fontweight="bold", fontsize=10)

    # Right: zero_ratio (degeneration indicator)
    zero_ratio = [0.172, 0.172, 0.171, 1.000]
    bars2 = ax2.bar(labels, zero_ratio, color=colors, edgecolor="#e0e0ff", linewidth=1.4, width=0.6)
    bars2[0].set_linewidth(2.4)
    ax2.axhline(0.90, color="#ff5252", lw=1.5, ls="--", alpha=0.8)
    ax2.text(0.5, 0.91, "collapse threshold 90%", color="#ff8a8a", fontsize=9, ha="center")
    ax2.set_ylabel("OOD zero_ratio (fraction predicting 0)")
    ax2.set_title("Degeneration Check: all-zero collapse ratio", fontweight="bold")
    ax2.set_ylim(0, 1.15)
    for b, v in zip(bars2, zero_ratio):
        flag = "[OK]" if v < 0.9 else "[FAIL]"
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}\n{flag}", ha="center",
                 color="#8aff8a" if v < 0.9 else "#ff8a8a", fontweight="bold", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig7_ablation.png")


def main():
    print("=" * 60)
    print("  Regenerate figures (ASCII-safe) + add fig7 ablation")
    print("=" * 60)
    setup_style()
    data = collect_all()
    for disp, info in data.items():
        n_ok = sum(1 for r in info["rows"] if r["ood"])
        print(f"  {disp:28s}: {n_ok}/5 seeds OOD data")
    print("\nGenerating figures...")
    fig1_training(data)
    fig2_ood_curve(data)
    fig3_ood_decay(data)
    fig4_param_efficiency(data)
    fig5_collapse(data)
    fig6_seed_variance(data)
    fig7_ablation()
    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
