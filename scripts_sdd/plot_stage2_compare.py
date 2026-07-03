"""Stage 2 SDD: 3k vs 10k changed_acc_curve 对比图 (宇宙深色风).
展示训练步数对 OOD 衰减的影响 + ch@100 边界异常深坑 + window decay.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans',
                                    'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 宇宙深色风配色
BG = '#0a0a1a'
PANEL = '#12122a'
GRID = '#2a2a4a'
TEXT = '#e8e8ff'
C_3K = '#ff6b9d'    # 粉红 (3k, 衰减)
C_10K = '#4ecdc4'   # 青绿 (10k, 增强)
C_ANOM = '#ffd93d'  # 金 (异常点)
C_WIN = '#a78bfa'   # 紫 (window)

# 加载数据
D3 = json.load(open(r'e:\three_chain_v3\results_stage2\sdd_30m_seed0_3k\ood_metrics.json'))
D10 = json.load(open(r'e:\three_chain_v3\results_stage2\sdd_30m_seed0_10k\ood_metrics.json'))

c3 = D3['changed_acc_curve']
c10 = D10['changed_acc_curve']
T = sorted(int(k) for k in c3.keys())
y3 = np.array([c3[str(t)] for t in T])
y10 = np.array([c10[str(t)] for t in T])

# window 均值
def window_avg(arr, idx, half=5):
    s, n = 0, 0
    for i in range(max(0, idx-half), min(len(arr), idx+half+1)):
        s += arr[i]; n += 1
    return s/n if n else None

idx100 = T.index(100)
idx150 = T.index(150)
w100_3k = window_avg(y3, idx100)
w150_3k = window_avg(y3, idx150)
w100_10k = window_avg(y10, idx100)
w150_10k = window_avg(y10, idx150)

# ========== 主图: 曲线对比 ==========
fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
ax.set_facecolor(PANEL)

# 渐变背景区域 (训练区 vs OOD 区)
ax.axvspan(0, 100, alpha=0.08, color='#4ecdc4', zorder=0)
ax.axvspan(100, 150, alpha=0.12, color='#ff6b9d', zorder=0)
ax.text(50, 0.32, '训练区 (T≤100)', color=C_10K, fontsize=10,
        ha='center', alpha=0.7, style='italic')
ax.text(125, 0.32, 'OOD 外推区 (T>100)', color=C_3K, fontsize=10,
        ha='center', alpha=0.7, style='italic')

# 曲线 (smoothed ±3 邻域均值减少抖动, 但保留 ch@100 原始点)
def smooth(arr, half=3):
    out = arr.copy()
    for i in range(len(arr)):
        s, n = 0, 0
        for j in range(max(0,i-half), min(len(arr),i+half+1)):
            s += arr[j]; n += 1
        out[i] = s/n
    return out

ax.plot(T, smooth(y3), color=C_3K, linewidth=2.2, alpha=0.85,
        label=f'SSM 3k 步 (单点 decay = {D3["ood_decay_pct"]*100:+.1f}%)',
        zorder=3)
ax.plot(T, smooth(y10), color=C_10K, linewidth=2.5, alpha=0.95,
        label=f'SSM 10k 步 (单点 decay = {D10["ood_decay_pct"]*100:+.1f}%)',
        zorder=4)

# 标注 ch@100 异常深坑
ax.scatter([100], [c10['100']], s=120, color=C_ANOM, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate('ch@100 异常深坑\n(训练 max_T 边界效应)\nch@99=0.256 → ch@100=0.178 → ch@101=0.254',
            xy=(100, c10['100']), xytext=(115, 0.10),
            color=C_ANOM, fontsize=8.5, ha='left',
            arrowprops=dict(arrowstyle='->', color=C_ANOM, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG,
                      edgecolor=C_ANOM, alpha=0.85))

# 标注 window decay
ax.scatter([100, 100], [w100_3k, w100_10k], s=60, color=C_WIN,
           marker='D', zorder=5, edgecolor='white', linewidth=1)
ax.scatter([150, 150], [w150_3k, w150_10k], s=60, color=C_WIN,
           marker='D', zorder=5, edgecolor='white', linewidth=1)

# 竖线
ax.axvline(100, color=GRID, linestyle='--', linewidth=1, alpha=0.6)
ax.axhline(0.0625, color='#666', linestyle=':', linewidth=1, alpha=0.5)
ax.text(3, 0.067, '随机基线 (1/16=0.0625)', color='#888', fontsize=8)

ax.set_xlabel('时间步 T', color=TEXT, fontsize=12)
ax.set_ylabel('changed_acc (S_t≠S_0 的 cell 预测准确率)', color=TEXT, fontsize=11)
ax.set_title('Stage 2 SDD 真实数据: 训练步数对 OOD 长程外推的影响\n'
             'bookstore 场景 · 30M SSM · T_train=100 → T_eval=150',
             color=TEXT, fontsize=13, pad=12)
ax.legend(loc='upper left', facecolor=BG, edgecolor=GRID,
          labelcolor=TEXT, fontsize=10)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.grid(True, alpha=0.2, color=GRID)
ax.set_xlim(0, 152)
ax.set_ylim(0, 0.35)

# 底部说明
fig.text(0.5, 0.01,
         '◆ 关键发现: 3k 步单点 decay=-26.5% (误判为衰减) → 10k 步充分训练后 window decay=+11.7% (长程增强)\n'
         '◆ ch@100 单点为训练边界异常, 必须用 window-smoothed decay (±5 邻域均值) 才稳健\n'
         '◆ 结论: SSM 不退化机制在 SDD 真实数据上依然成立 (与 GridWorld 100M→1.13B 一致)',
         color=C_ANOM, fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                   edgecolor=C_ANOM, alpha=0.9))

plt.tight_layout(rect=[0, 0.08, 1, 1])
OUT = r'e:\three_chain_v3\paper\figures\stage2_sdd_3k_vs_10k.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, facecolor=BG, bbox_inches='tight')
print(f"✓ 已保存: {OUT}")

# ========== 副图: window decay 对比柱状图 ==========
fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=BG)
ax2.set_facecolor(PANEL)

labels = ['3k 步\n(未充分训练)', '10k 步\n(充分训练)']
single = [D3['ood_decay_pct']*100, D10['ood_decay_pct']*100]
window = [(w150_3k-w100_3k)/w100_3k*100 if w100_3k else 0,
          (w150_10k-w100_10k)/w100_10k*100 if w100_10k else 0]

x = np.arange(len(labels))
w = 0.35
b1 = ax2.bar(x - w/2, single, w, color=C_3K, alpha=0.85,
             label='单点 decay (ch@100 → ch@150)', edgecolor='white')
b2 = ax2.bar(x + w/2, window, w, color=C_10K, alpha=0.95,
             label='window decay (±5 邻域均值)', edgecolor='white')

# ±2% 不退化阈值带
ax2.axhspan(-2, 2, alpha=0.15, color=C_WIN, zorder=0)
ax2.axhline(0, color=TEXT, linewidth=0.8, alpha=0.5)

# 数值标签
for bars, vals in [(b1, single), (b2, window)]:
    for bar, v in zip(bars, vals):
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2,
                 h + (1.5 if h >= 0 else -3.5),
                 f'{v:+.1f}%', ha='center',
                 color=TEXT if h >= 0 else C_3K, fontsize=11,
                 fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, color=TEXT, fontsize=11)
ax2.set_ylabel('OOD decay % (正值=长程增强, 负值=衰减)',
               color=TEXT, fontsize=11)
ax2.set_title('Stage 2 SDD: 训练充分性对 OOD decay 的影响\n'
              '(±2% 紫带为不退化阈值)',
              color=TEXT, fontsize=12, pad=10)
ax2.legend(loc='upper left', facecolor=BG, edgecolor=GRID,
           labelcolor=TEXT, fontsize=9)
ax2.tick_params(colors=TEXT)
for spine in ax2.spines.values():
    spine.set_color(GRID)
ax2.grid(True, axis='y', alpha=0.2, color=GRID)
ax2.set_ylim(-35, 60)

fig2.text(0.5, 0.02,
          '◆ 3k 单点 -26.5% 是未充分训练假象 + ch@100 边界噪声\n'
          '◆ 10k 充分训练后: 单点 +53.2% (ch@100 坑放大) / window +11.7% (稳健)\n'
          '◆ 真实数据上 SSM 仍展现长程增强 (非退化)',
          color=C_ANOM, fontsize=9, ha='center', style='italic')

plt.tight_layout(rect=[0, 0.08, 1, 1])
OUT2 = r'e:\three_chain_v3\paper\figures\stage2_decay_3k_vs_10k.png'
plt.savefig(OUT2, dpi=150, facecolor=BG, bbox_inches='tight')
print(f"✓ 已保存: {OUT2}")
print("\n=== 图表生成完成 ===")
