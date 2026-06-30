"""打印 BP v2 OOD 关键指标。"""
import json
from pathlib import Path

root = Path("results_wsl")
d = json.loads((root / "run_three_chain_bp_v2_seed0/ood_metrics.json").read_text(encoding="utf-8"))
c = d["changed_acc_curve"]
print("=== BP v2 OOD changed_acc 关键点 ===")
for t in [50, 100, 120, 150]:
    if str(t) in c:
        print(f"  ch@{t}: {c[str(t)]:.4f}")
a100, a150 = c.get("100", 0), c.get("150", 0)
dec = (a150 - a100) / a100 * 100 if a100 > 0 else 0
print(f"  decay (100->150): {dec:+.1f}%")
print(f"  acc_final(t=150): {d['acc_final']:.4f}")
