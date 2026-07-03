import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/run_three_chain_mamba3_seed0/ood_metrics.json"
with open(path) as f:
    d = json.load(f)

print("top-level keys:", list(d.keys()))
print("acc_final:", d.get("acc_final"))
print("changed_acc (overall):", d.get("changed_acc"))

ca = d.get("changed_acc_curve", {})
print("\nchanged_acc_curve: total points =", len(ca))
# 打印所有时间步的 changed_acc
for t in sorted(ca.keys(), key=lambda x: int(x)):
    print(f"  t={t}: {ca[t]:.4f}")
