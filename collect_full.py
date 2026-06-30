import sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

BASE = "E:/three_chain_v3/results_wsl/benchmark_multiseed"

print("=" * 72)
print("  FULL BENCHMARK: Standard + OOD Extrapolation")
print("=" * 72)

for model, label in [("three_chain", "Tri-Helix"), ("single_chain", "Single Mamba"), ("transformer", "Transformer")]:
    std_accs = []
    ood_accs = []
    ood_chs = []
    params = None
    
    for seed in [42, 123, 456, 789, 1024]:
        sf = f"{BASE}/{model}_seed{seed}/metrics.json"
        of = f"{BASE}/{model}_seed{seed}/ood_metrics.json"
        try:
            sd = json.load(open(sf))
            od = json.load(open(of))
            std_accs.append(sd.get("acc_final", 0))
            ood_accs.append(od.get("acc_final", 0))
            ood_chs.append(od.get("changed_acc", 0))
            if params is None: params = sd.get("params", "?")
        except: pass
    
    std_accs = np.array(std_accs)
    ood_accs = np.array(ood_accs)
    ood_chs = np.array(ood_chs)
    
    print(f"\n### {label} ({params} params)")
    print(f"  Standard acc (T=100):    {np.mean(std_accs):.4f} +/- {np.std(std_accs, ddof=1):.4f}")
    print(f"  OOD acc (T=150):         {np.mean(ood_accs):.4f} +/- {np.std(ood_accs, ddof=1):.4f}")
    print(f"  OOD ch_acc (T=150):      {np.mean(ood_chs):.4f} +/- {np.std(ood_chs, ddof=1):.4f}")
    # Stability = inverse of coefficient of variation
    cv = np.std(ood_accs, ddof=1) / max(np.mean(ood_accs), 0.001)
    print(f"  OOD stability (1/CV):    {1/max(cv,0.001):.1f}")
    print(f"  Seed OOD accs:           {[round(a,4) for a in ood_accs.tolist()]}")

print()
print("=" * 72)
print("  VERDICT")
print("=" * 72)
