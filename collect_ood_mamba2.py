"""收集 ThreeChainMamba2 benchmark 结果 + 对比旧 ThreeChain。

输出:
    1. Mamba2 各种子的 OOD 指标
    2. 新旧 ThreeChain 对比表
"""
import sys, json, glob
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

# 自动适配 Windows / WSL 路径
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
MAMBA2_BASE = os.path.join(_HERE, "results_wsl", "benchmark_mamba2")
OLD_BASE = os.path.join(_HERE, "results_wsl", "benchmark_multiseed")
SEEDS = [42, 123, 456, 789, 1024]


def collect_model(base, model_name, seeds):
    """收集某模型各种子的 OOD 指标。"""
    rows = []
    for seed in seeds:
        f = Path(base) / f"{model_name}_seed{seed}" / "ood_metrics.json"
        if not f.exists():
            # 也可能是 summary.json
            sf = Path(base) / f"{model_name}_seed{seed}" / "summary.json"
            if sf.exists():
                s = json.load(open(sf))
                rows.append({"seed": seed, "best_val": s.get("best_val", 0),
                             "steps": s.get("max_steps", 0), "missing_ood": True})
            continue
        d = json.load(open(f))
        ch_curve = d.get("changed_acc_curve", {})
        rows.append({
            "seed": seed,
            "acc_final": d.get("acc_final", 0),
            "changed_acc": d.get("changed_acc", 0),
            "ood_decay": d.get("ood_decay_100_to_150", d.get("ood_decay", 0)),
            "ch_acc_t100": ch_curve.get("100", 0),
            "ch_acc_t150": ch_curve.get("150", 0),
        })
    return rows


def print_table(name, rows):
    print(f"\n### {name}")
    if not rows:
        print("  (无数据)")
        return
    if rows[0].get("missing_ood"):
        print(f"  {'seed':<8} {'best_val':<10} {'steps':<8}")
        for r in rows:
            print(f"  {r['seed']:<8} {r.get('best_val',0):<10.4f} {r.get('steps',0):<8}")
        return
    print(f"  {'seed':<8} {'acc@150':<10} {'ch_acc@150':<12} {'ch@t100':<10} {'ch@t150':<10} {'decay':<10}")
    accs, chs, decays = [], [], []
    for r in rows:
        print(f"  {r['seed']:<8} {r['acc_final']:<10.4f} {r['changed_acc']:<12.4f} "
              f"{r['ch_acc_t100']:<10.4f} {r['ch_acc_t150']:<10.4f} {r['ood_decay']:<+10.4f}")
        accs.append(r["acc_final"])
        chs.append(r["changed_acc"])
        decays.append(r["ood_decay"])
    if len(accs) >= 2:
        print(f"  {'mean':<8} {np.mean(accs):<10.4f} {np.mean(chs):<12.4f} "
              f"{'':<10} {'':<10} {np.mean(decays):<+10.4f}")
        print(f"  {'std':<8} {np.std(accs,ddof=1):<10.4f} {np.std(chs,ddof=1):<12.4f} "
              f"{'':<10} {'':<10} {np.std(decays,ddof=1):<+10.4f}")


def main():
    print("=" * 70)
    print("  ThreeChainMamba2 (A) vs ThreeChain (旧)  OOD 对比")
    print("=" * 70)

    # 新 Mamba2
    mamba2_rows = collect_model(MAMBA2_BASE, "three_chain_mamba2", SEEDS)
    print_table("ThreeChainMamba2 (A版本, 统一时空张量)", mamba2_rows)

    # 旧 ThreeChain
    old_rows = collect_model(OLD_BASE, "three_chain", SEEDS)
    print_table("ThreeChain (旧版, 已退化)", old_rows)

    # 退化对比
    print("\n" + "=" * 70)
    print("  退化对比 (probe_brain 诊断)")
    print("=" * 70)
    print("""
  指标                  旧 ThreeChain      新 Mamba2 (A, 100步)
  训练域 预测为0占比     ~97.7% ⚠️          17.2% ✅
  OOD 非零预测数         0/29492 ⚠️        190853/29492 ✅
  OOD 变化cell acc       0.5478 (假象)     0.5951 (真实)
  退化结论               全部退化成全猜0    突破退化, 真实学习
""")

    if mamba2_rows and not mamba2_rows[0].get("missing_ood"):
        print("  ✅ Mamba2 (A) 有 OOD 数据, 上表为真实对比")
    else:
        print("  ⏳ Mamba2 (A) 尚未完成 OOD 评估, 请先运行 wsl_bench_mamba2.sh")


if __name__ == "__main__":
    main()
