"""打印 Mamba3 vs Mamba2 OOD 对照表（试验出真知）"""
import json
from pathlib import Path

files = {
    "Mamba3": "results/run_three_chain_mamba3_seed0/ood_metrics.json",
    "Mamba2": "results/run_three_chain_mamba2_100step_m3subset_seed0/ood_metrics.json",
}
summaries = {
    "Mamba3": "results/run_three_chain_mamba3_seed0/log.jsonl",
    "Mamba2": "results/run_three_chain_mamba2_100step_m3subset_seed0/log.jsonl",
}

print("=" * 70)
print("Mamba3 vs Mamba2 对照实验（100 步，同 N≤8 数据子集，同 seed=0）")
print("=" * 70)

for name, path in files.items():
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"\n【{name}】")
    print(f"  acc_final (t=150):     {m['acc_final']:.4f}")
    print(f"  changed_acc (t=150):   {m['changed_acc']:.4f}")
    print(f"  OOD 衰减 (t=100→150): {m.get('ood_decay_100_to_150', 0):+.4f} "
          f"({m.get('ood_decay_pct', 0)*100:+.1f}%)")
    curve = m["changed_acc_curve"]
    for t in ["50", "100", "120", "150"]:
        if t in curve:
            print(f"    changed_acc@t={t}:   {curve[t]:.4f}")

print("\n" + "=" * 70)
print("训练指标对照（100 步日志）")
print("=" * 70)
for name, path in summaries.items():
    print(f"\n【{name}】  (取最后一行 step=100)")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    last = json.loads(lines[-1])
    print(f"  step={last.get('step')}  loss={last.get('loss', 0):.4f}  "
          f"ch_acc={last.get('changed_acc', 0):.4f}  "
          f"elapsed={last.get('elapsed', 0):.0f}s")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print("""
  ✅ Mamba2: OOD 衰减 +0.7%（基本零衰减），ch_acc=0.669，训练 49s
  ❌ Mamba3: OOD 衰减 -11.5%（明显退化），ch_acc=0.491，训练 1031s

  Mamba3 在所有维度都劣于 Mamba2：
    1. OOD 长程外推：反而退化（理论上的 trapezoidal+RoPE 优势没兑现）
    2. 学习速度：慢 27%（0.491 vs 0.669）
    3. 训练时间：慢 21x（17 min vs 49 s）

  按用户原则"试验不行就不用"——Mamba3 不替换 Mamba2 backbone。
  保留 mamba3_ref.py + three_chain_mamba3.py 作为论文 ablation 用：
  证明"更复杂的 SSM 不一定更适合三链长程外推任务"。
""")
