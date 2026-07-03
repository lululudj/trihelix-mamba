"""30M 500步 decay 深度分析: 单点 vs 窗口平滑, 判断 -9.81% 是真退化还是 ch@100 峰值假象"""
import json
import statistics
from pathlib import Path

BASE = Path(r"e:\three_chain_v3\results_cloud")
seeds = [0, 1, 2, 3, 4]


def window_avg(curve, center, half=5):
    """以 center 为中心取 [center-half, center+half] 窗口均值"""
    vals = []
    for t in range(center - half, center + half + 1):
        v = curve.get(str(t))
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


print("30M 500步 decay 深度分析 (单点 vs 窗口平滑)")
print("=" * 90)
print(f"{'seed':<6}{'ch@100单点':<12}{'ch@100窗口':<12}{'ch@150单点':<12}{'ch@150窗口':<12}"
      f"{'decay单点':<12}{'decay窗口':<12}{'判定':<10}")
print("-" * 90)

sp_decays = []   # single-point
win_decays = []  # window-smoothed
for s in seeds:
    p = BASE / f"run_30m_seed{s}_500" / "ood_metrics.json"
    d = json.load(open(p, encoding='utf-8'))
    c = d.get('changed_acc_curve', {})
    ch100_sp = c.get('100')
    ch150_sp = c.get('150')
    ch100_win = window_avg(c, 100, 5)
    ch150_win = window_avg(c, 150, 5)
    if ch100_sp and ch150_sp:
        decay_sp = (ch150_sp - ch100_sp) / ch100_sp * 100
        sp_decays.append(decay_sp)
    else:
        decay_sp = None
    if ch100_win and ch150_win:
        decay_win = (ch150_win - ch100_win) / ch100_win * 100
        win_decays.append(decay_win)
    else:
        decay_win = None
    judge = "✓" if (decay_win is not None and -2 <= decay_win <= 2) else ("~" if (decay_win is not None and -3 <= decay_win <= 3) else "✗")
    print(f"{s:<6}{ch100_sp:<12.4f}{ch100_win:<12.4f}{ch150_sp:<12.4f}{ch150_win:<12.4f}"
          f"{decay_sp:<+12.2f}{decay_win:<+12.2f}{judge:<10}")

print()
print("=" * 90)
print("汇总统计:")
print(f"  单点 decay:  median={statistics.median(sp_decays):+.2f}%  mean={statistics.mean(sp_decays):+.2f}%  "
      f"min={min(sp_decays):+.2f}%  max={max(sp_decays):+.2f}%  spread={max(sp_decays)-min(sp_decays):.2f}%")
print(f"  窗口 decay:  median={statistics.median(win_decays):+.2f}%  mean={statistics.mean(win_decays):+.2f}%  "
      f"min={min(win_decays):+.2f}%  max={max(win_decays):+.2f}%  spread={max(win_decays)-min(win_decays):.2f}%")
print()
sp_in = sum(1 for d in sp_decays if -2 <= d <= 2)
win_in = sum(1 for d in win_decays if -2 <= d <= 2)
print(f"  ±2% 内:  单点 {sp_in}/{len(sp_decays)}   窗口 {win_in}/{len(win_decays)}")
print()
print("结论判定:")
if all(-2 <= d <= 2 for d in win_decays):
    print("  ✓ 窗口平滑后全部 ±2% 内 → -9.81% 是 ch@100 局部峰值假象, 30M 实际不退化")
    print("  → 可诚实报告: decay 指标在小尺度曲线噪声下应取窗口均值")
elif win_in > sp_in:
    print(f"  ~ 窗口平滑把 {win_in - sp_in} 个 seed 拉回 ±2% 内, 但仍有离群")
    print("  → 30M 中位 decay 在 ±2% 内, 但方差大, 不能作为 '不退化' 的硬数据点")
else:
    print("  ✗ 窗口平滑也救不回来 → 30M 确实不稳/退化")
