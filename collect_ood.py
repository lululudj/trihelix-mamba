import sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

BASE = "E:/three_chain_v3/results_wsl/benchmark_multiseed"

print("=" * 70)
print("  OOD EXTRAPOLATION RESULTS (T=100 -> T=150)")
print("=" * 70)

for model, label in [("three_chain", "Tri-Helix"), ("single_chain", "Single Mamba"), ("transformer", "Transformer")]:
    decays = []
    accs_t150 = []
    chs_t150 = []
    for seed in [42, 123, 456, 789, 1024]:
        f = f"{BASE}/{model}_seed{seed}/ood_metrics.json"
        try:
            d = json.load(open(f))
            # eval_ood.py 输出 ood_decay_100_to_150，旧字段名 ood_decay 兼容
            decays.append(d.get("ood_decay_100_to_150", d.get("ood_decay", 0)))
            accs_t150.append(d.get("acc_final", 0))
            chs_t150.append(d.get("changed_acc", 0))
        except:
            pass
    
    if len(decays) >= 3:
        decays = np.array(decays)
        mean_d = np.mean(decays) * 100
        std_d = np.std(decays, ddof=1) * 100
        print(f"\n### {label}")
        print(f"    OOD decay:    {mean_d:+.1f}% +/- {std_d:.1f}%")
        print(f"    acc@T=150:     {np.mean(accs_t150):.4f} +/- {np.std(accs_t150, ddof=1):.4f}")
        print(f"    ch_acc@T=150:  {np.mean(chs_t150):.4f} +/- {np.std(chs_t150, ddof=1):.4f}")
        seed_strs = [str(round(d*100, 1)) + "%" for d in decays]
        print(f"    seeds decay:   {seed_strs}")

print()
print("Lower OOD decay = better extrapolation. 0% = perfect.")
