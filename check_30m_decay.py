"""快速查看 30M 500 步多 seed 补跑的 decay 结果"""
import json
from pathlib import Path

BASE = Path(r"e:\three_chain_v3\results_cloud")
seeds = [0, 1, 2, 3, 4]
print("30M 500 步多 seed 补跑 decay 结果:")
print("=" * 70)
print(f"{'seed':<6} {'ch@100':<10} {'ch@150':<10} {'decay':<10} {'val_ch':<10}")
print("-" * 70)

decays = []
for s in seeds:
    p = BASE / f"run_30m_seed{s}_500" / "ood_metrics.json"
    if not p.exists():
        print(f"{s:<6} (尚未下载)")
        continue
    d = json.load(open(p, encoding='utf-8'))
    c = d.get('changed_acc_curve', {})
    ch100 = c.get('100')
    ch150 = c.get('150')
    if ch100 is None or ch150 is None or ch100 == 0:
        print(f"{s:<6} (曲线数据不全)")
        continue
    decay = (ch150 - ch100) / ch100 * 100
    decays.append(decay)
    val_ch = d.get('changed_acc', '?')
    print(f"{s:<6} {ch100:<10.4f} {ch150:<10.4f} {decay:<+10.2f}% {val_ch if isinstance(val_ch, str) else f'{val_ch:.4f}'}")

print()
if len(decays) >= 2:
    spread = max(decays) - min(decays)
    print(f"spread = {spread*100:.2f}%")
    in_band = sum(1 for d in decays if -2 <= d <= 2)
    print(f"±2% 内: {in_band}/{len(decays)}")
    if in_band == len(decays):
        print("✓ 30M 500步全部在 ±2% 内, '30M→1B 不退化' 主张恢复!")
    else:
        print("⚠ 部分 seed 超 ±2%, 需进一步分析")
