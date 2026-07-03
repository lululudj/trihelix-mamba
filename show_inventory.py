"""列出本地 results_cloud 中所有已下载的实验结果"""
import json
from pathlib import Path

LOCAL_DIR = Path(r"e:\three_chain_v3\results_cloud")

print(f"本地结果目录: {LOCAL_DIR}\n")
print(f"{'实验':<40} {'ch@100':>8} {'ch@150':>8} {'decay':>8}  状态")
print("-" * 80)

exps = []
for exp_dir in sorted(LOCAL_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    ood_file = exp_dir / "ood_metrics.json"
    summary_file = exp_dir / "summary.json"
    if ood_file.exists():
        try:
            m = json.loads(ood_file.read_text(encoding="utf-8"))
            ch = m.get("changed_acc_curve", {})
            a100 = ch.get("100")
            a150 = ch.get("150")
            if a100 is not None and a150 is not None:
                decay = (a150 - a100) / a100 * 100 if a100 > 0 else 0
                exps.append((exp_dir.name, a100, a150, decay, "✅ 有OOD结果"))
            else:
                exps.append((exp_dir.name, None, None, None, "⚠ OOD无曲线"))
        except Exception as e:
            exps.append((exp_dir.name, None, None, None, f"❌ 解析错误"))
    elif summary_file.exists():
        exps.append((exp_dir.name, None, None, None, "⚠ 仅有summary"))
    else:
        files = list(exp_dir.glob("*"))
        if files:
            exps.append((exp_dir.name, None, None, None, f"⚠ {len(files)}个文件无ood_metrics"))
        else:
            exps.append((exp_dir.name, None, None, None, "❌ 空目录"))

for name, a100, a150, decay, status in exps:
    if a100 is not None:
        print(f"{name:<40} {a100:>8.4f} {a150:>8.4f} {decay:>+7.1f}%  {status}")
    else:
        print(f"{name:<40} {'?':>8} {'?':>8} {'?':>8}  {status}")

# 统计
done = [e for e in exps if e[3] is not None]
print(f"\n总计: {len(exps)} 个实验目录, {len(done)} 个有完整 OOD 结果")

# 实验矩阵
print("\n=== 实验矩阵 (按参数量分组) ===")
groups = {}
for name, a100, a150, decay, _ in exps:
    if "30m" in name and "100m" not in name:
        key = "30M (27M params)"
    elif "100m_hta" in name:
        key = "100M HTA (72M)"
    elif "100m" in name and "1000m" not in name:
        key = "100M (72M params)"
    elif "300m" in name:
        key = "300M (182M params)"
    elif "700m" in name:
        key = "700M (~728M params)"
    elif "1000m" in name:
        key = "1000M (~1.1B params)"
    else:
        key = "baseline (3.26M)"
    groups.setdefault(key, []).append((name, a100, a150, decay))

for key in ["baseline (3.26M)", "30M (27M params)", "100M (72M params)", "100M HTA (72M)", "300M (182M params)", "700M (~728M params)", "1000M (~1.1B params)"]:
    if key not in groups:
        continue
    items = groups[key]
    done_items = [i for i in items if i[3] is not None]
    if done_items:
        decays = [i[3] for i in done_items]
        avg = sum(decays) / len(decays)
        print(f"\n{key}: {len(done_items)}/{len(items)} 完成, 平均 decay={avg:+.1f}%")
        for name, a100, a150, decay in items:
            if decay is not None:
                print(f"  {name:<40} {decay:+.1f}%")
            else:
                print(f"  {name:<40} (未完成)")
    else:
        print(f"\n{key}: 0/{len(items)} 完成")
        for name, _, _, _ in items:
            print(f"  {name:<40} (未完成)")
