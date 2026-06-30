"""对比三个 ThreeChain-family 模型的 OOD metrics，看是否字节相同。"""
import json
from pathlib import Path

root = Path("results_wsl")
files = {
    "three_chain": root / "run_three_chain_seed0/ood_metrics.json",
    "bp_v1": root / "run_three_chain_bp_seed0/ood_metrics.json",
    "bp_v2": root / "run_three_chain_bp_v2_seed0/ood_metrics.json",
    "single_chain": root / "run_single_chain_seed0/ood_metrics.json",
}

data = {}
for name, p in files.items():
    if p.exists():
        data[name] = json.loads(p.read_text(encoding="utf-8"))
        print(f"{name}: {p}")
    else:
        print(f"{name}: MISSING")

print("\n=== acc_final 对比 ===")
for name, d in data.items():
    print(f"  {name}: acc_final={d['acc_final']:.6f}, n_samples={d.get('n_samples')}")

print("\n=== changed_acc_curve 关键点对比 ===")
print(f"{'model':<16} {'ch@50':>10} {'ch@100':>10} {'ch@120':>10} {'ch@150':>10}")
for name, d in data.items():
    c = d["changed_acc_curve"]
    row = f"{name:<16}"
    for t in [50, 100, 120, 150]:
        v = c.get(str(t), float("nan"))
        row += f" {v:>10.6f}"
    print(row)

print("\n=== acc_curve t=1,50,100,150 对比（看是否完全相同）===")
print(f"{'model':<16} {'acc@1':>10} {'acc@50':>10} {'acc@100':>10} {'acc@150':>10}")
for name, d in data.items():
    c = d["acc_curve"]
    row = f"{name:<16}"
    for t in [1, 50, 100, 150]:
        v = c.get(str(t), float("nan"))
        row += f" {v:>10.6f}"
    print(row)

# 字节级对比：three_chain vs bp_v2 的 acc_curve 是否完全相同
if "three_chain" in data and "bp_v2" in data:
    tc_curve = data["three_chain"]["acc_curve"]
    bp_curve = data["bp_v2"]["acc_curve"]
    same = all(abs(tc_curve.get(k, -1) - bp_curve.get(k, -1)) < 1e-9 for k in set(tc_curve) | set(bp_curve))
    print(f"\nthree_chain vs bp_v2 acc_curve 完全相同: {same}")
    if not same:
        diffs = [(k, tc_curve.get(k), bp_curve.get(k)) for k in sorted(set(tc_curve) | set(bp_curve), key=int)
                 if abs(tc_curve.get(k, -1) - bp_curve.get(k, -1)) >= 1e-9]
        print(f"  差异点数: {len(diffs)}")
        for k, a, b in diffs[:5]:
            print(f"    t={k}: three_chain={a:.6f} vs bp_v2={b:.6f}")
