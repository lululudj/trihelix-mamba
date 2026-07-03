"""阶段 3 任务 3.1: Hidden state 可视化 (读 probe .npz 画图)

读 results_stage3/probe_*.npz, 画 4 张图 (宇宙深色风):
  Fig A: 三链 norm 随 t 演化 (最后层, 多 checkpoint 对照)
  Fig B: 三链间余弦相似度随 t 演化 (信息冗余/分化)
  Fig C: 第一层 vs 最后层 norm 对照 (深度演化效应)
  Fig D: 100步 vs 1000步训练对照 (训练对稳定性的影响)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROBE_DIR = Path("results_stage3")
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 宇宙深色风 (与 generate_figures.py 一致)
plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = '#0a0a1f'
mpl.rcParams['axes.facecolor'] = '#0a0a1f'
mpl.rcParams['savefig.facecolor'] = '#0a0a1f'
mpl.rcParams['axes.edgecolor'] = '#5555aa'
mpl.rcParams['axes.labelcolor'] = '#e0e0ff'
mpl.rcParams['xtick.color'] = '#aaaacc'
mpl.rcParams['ytick.color'] = '#aaaacc'
mpl.rcParams['text.color'] = '#e0e0ff'
mpl.rcParams['axes.titlecolor'] = '#e0e0ff'
mpl.rcParams['grid.color'] = '#333366'
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 三链配色 (蓝=空间, 紫=时间, 粉=因果)
COLOR_S = '#5b8cff'   # 空间链 蓝
COLOR_T = '#a866ff'   # 时间链 紫
COLOR_C = '#ff5b8c'   # 因果链 粉
COLOR_BAND = '#2a2a5a'
COLOR_TRAIN = '#ffaa55'  # 训练上限标记 金


def load_probe(path):
    """读 .npz 返回 dict (把 numpy 0-d/object array 转回 Python 原生类型)."""
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        # object array (dict saved as 1-elem object array) → .item() 取 Python dict
        # 0-d scalar/string array → .item() 取 Python int/float/str
        if v.dtype == object or v.ndim == 0:
            out[k] = v.item() if v.size == 1 else v.item()
        else:
            out[k] = v
    return out


def get_curve(d, key):
    """从 probe dict 取 {str(t): val} 转 (ts, vals) 数组."""
    curve = d[key]
    ts = sorted(int(k) for k in curve.keys())
    vals = [curve[str(t)] for t in ts]
    return np.array(ts), np.array(vals)


# ============================================================
# Fig A: 三链 norm 随 t 演化 (最后层, 多 checkpoint 对照)
# ============================================================
def fig_a_norm_evolution(probes):
    """三链 norm 随 t: 每个 checkpoint 一行, 三列 (s/t/c)."""
    n = len(probes)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n), squeeze=False)

    for row, (label, d) in enumerate(probes.items()):
        for col, (chain, color, name) in enumerate([
            ("norm_s_last", COLOR_S, "空间链 h_s"),
            ("norm_t_last", COLOR_T, "时间链 h_t"),
            ("norm_c_last", COLOR_C, "因果链 h_c"),
        ]):
            ax = axes[row][col]
            ts, vals = get_curve(d, chain)
            ax.plot(ts, vals, color=color, linewidth=2, alpha=0.9)
            ax.axvline(100, color=COLOR_TRAIN, linewidth=1.5, linestyle='--', alpha=0.7,
                       label='t=100 训练上限')
            ax.axhspan(vals[max(0, np.searchsorted(ts, 100)) - 1] * 0.95,
                       vals[max(0, np.searchsorted(ts, 100)) - 1] * 1.05,
                       color=COLOR_BAND, alpha=0.3)

            # 标注 OOD 区 decay
            idx100 = np.searchsorted(ts, 100)
            idx_final = len(vals) - 1
            if idx100 < len(vals) and idx_final > idx100:
                decay = (vals[idx_final] - vals[idx100]) / vals[idx100] * 100
                ax.annotate(f'OOD 变化\n{decay:+.1f}%',
                            xy=(ts[idx_final], vals[idx_final]),
                            xytext=(ts[idx_final] * 0.7, vals[idx_final] * 1.15),
                            color='#5bffaa' if abs(decay) < 5 else COLOR_TRAIN,
                            fontsize=9, ha='center',
                            arrowprops=dict(arrowstyle='->', color='#5bffaa', lw=1))

            ax.set_title(f'{name} ({label})', fontsize=10)
            ax.set_xlabel('时间步 t')
            ax.set_ylabel('L2 norm')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a3a',
                          edgecolor='#5555aa', framealpha=0.85)

    plt.suptitle('阶段 3.1: 三链 hidden state norm 随时间演化\n'
                 '(OOD 区 t>100 稳定性 → 不退化机制)',
                 fontsize=12, color='#e0e0ff', y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'stage3_hidden_probes_norm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out}", flush=True)


# ============================================================
# Fig B: 三链间余弦相似度随 t 演化
# ============================================================
def fig_b_cosine_similarity(probes):
    """三链间余弦相似度: 每个 checkpoint 一图."""
    n = len(probes)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, (label, d) in enumerate(probes.items()):
        ax = axes[0][i]
        for key, color, name in [
            ("cos_st_last", COLOR_S, "空间 vs 时间"),
            ("cos_sc_last", COLOR_T, "空间 vs 因果"),
            ("cos_tc_last", COLOR_C, "时间 vs 因果"),
        ]:
            ts, vals = get_curve(d, key)
            ax.plot(ts, vals, color=color, linewidth=2, alpha=0.85, label=name)

        ax.axvline(100, color=COLOR_TRAIN, linewidth=1.5, linestyle='--', alpha=0.7,
                   label='t=100 训练上限')
        ax.axhline(0, color='#6666aa', linewidth=0.5, linestyle=':', alpha=0.4)
        ax.set_title(f'三链余弦相似度 ({label})\n(值低=信息分化, 值高=冗余)',
                     fontsize=10)
        ax.set_xlabel('时间步 t')
        ax.set_ylabel('余弦相似度')
        ax.set_ylim(-0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a3a',
                  edgecolor='#5555aa', framealpha=0.85)

    plt.suptitle('阶段 3.1: 三链间信息分化 (余弦相似度随 t)\n'
                 'OOD 区相似度稳定 → 三链维持互补分工',
                 fontsize=12, color='#e0e0ff', y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'stage3_hidden_probes_cosine.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out}", flush=True)


# ============================================================
# Fig C: 第一层 vs 最后层 norm 对照 (深度演化)
# ============================================================
def fig_c_depth_comparison(probes):
    """第一层 vs 最后层: 看三链在深度上的演化差异."""
    # 取第一个 checkpoint 做深度对照
    label, d = next(iter(probes.items()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for col, (chain_last, chain_l0, color, name) in enumerate([
        ("norm_s_last", "norm_s_layer0", COLOR_S, "空间链 h_s"),
        ("norm_t_last", "norm_t_layer0", COLOR_T, "时间链 h_t"),
        ("norm_c_last", "norm_c_layer0", COLOR_C, "因果链 h_c"),
    ]):
        ax = axes[col]
        ts1, v1 = get_curve(d, chain_l0)
        ts2, v2 = get_curve(d, chain_last)
        ax.plot(ts1, v1, color=color, linewidth=2, alpha=0.6, linestyle='--',
                label='第一层 (演化初期)')
        ax.plot(ts2, v2, color=color, linewidth=2.5, alpha=0.95,
                label='最后层 (最抽象)')
        ax.axvline(100, color=COLOR_TRAIN, linewidth=1.5, linestyle=':', alpha=0.6,
                   label='t=100 训练上限')
        ax.set_title(f'{name} 深度对照 ({label})', fontsize=10)
        ax.set_xlabel('时间步 t')
        ax.set_ylabel('L2 norm')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, facecolor='#1a1a3a',
                  edgecolor='#5555aa', framealpha=0.85)

    plt.suptitle('阶段 3.1: 第一层 vs 最后层 norm 对照\n'
                 '(深度演化对三链稳定性的影响)',
                 fontsize=12, color='#e0e0ff', y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'stage3_hidden_probes_depth.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out}", flush=True)


# ============================================================
# Fig D: 100步 vs 1000步训练对照
# ============================================================
def fig_d_training_effect(probes):
    """训练步数对三链稳定性的影响: 多 checkpoint 同链对比."""
    if len(probes) < 2:
        print("  [skip] Fig D 需要至少 2 个 checkpoint 对照")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#5b8cff', '#ff5b8c', '#5bffaa', '#ffaa55']

    for col, (chain_key, name) in enumerate([
        ("norm_s_last", "空间链 h_s"),
        ("norm_t_last", "时间链 h_t"),
        ("norm_c_last", "因果链 h_c"),
    ]):
        ax = axes[col]
        for i, (label, d) in enumerate(probes.items()):
            ts, vals = get_curve(d, chain_key)
            # 归一化到 t=100 的值 (看相对变化)
            idx100 = np.searchsorted(ts, 100)
            if idx100 < len(vals) and vals[idx100] > 0:
                vals_norm = vals / vals[idx100] * 100
            else:
                vals_norm = vals
            ax.plot(ts, vals_norm, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.9, label=label)

        ax.axvline(100, color=COLOR_TRAIN, linewidth=1.5, linestyle='--', alpha=0.7,
                   label='t=100 训练上限')
        ax.axhline(100, color='#6666aa', linewidth=0.5, linestyle=':', alpha=0.4)
        ax.set_title(f'{name} (归一化到 t=100)\nOOD 区偏离 100% = 不稳定', fontsize=10)
        ax.set_xlabel('时间步 t')
        ax.set_ylabel('norm (相对 t=100, %)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, facecolor='#1a1a3a',
                  edgecolor='#5555aa', framealpha=0.85)

    plt.suptitle('阶段 3.1: 训练步数对三链稳定性的影响\n'
                 '(1000步训练 → OOD 区更稳定)',
                 fontsize=12, color='#e0e0ff', y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'stage3_hidden_probes_training.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out}", flush=True)


def main():
    print("=== 阶段 3.1 hidden state 可视化 ===")
    # 自动发现所有 probe_*.npz
    probes = {}
    for p in sorted(PROBE_DIR.glob("probe_*.npz")):
        d = load_probe(p)
        label = d.get("label", p.stem)
        probes[label] = d
        print(f"  读: {p.name} (label={label}, n_samples={d.get('n_samples')})")

    if not probes:
        print("[错误] 没找到 probe_*.npz, 先跑 probe_hidden_states.py")
        sys.exit(1)

    print(f"\n共 {len(probes)} 个 checkpoint 探针结果, 画 4 张图...")
    fig_a_norm_evolution(probes)
    fig_b_cosine_similarity(probes)
    fig_c_depth_comparison(probes)
    fig_d_training_effect(probes)
    print(f"\n=== 完成, 图表在 {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
