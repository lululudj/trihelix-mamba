import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
MODELS = ["three_chain", "single_chain", "transformer"]
BASE = Path("E:/three_chain_v3/results_wsl/benchmark_multiseed")

print("=" * 60)
print("  MULTI-SEED BENCHMARK RESULTS")
print("=" * 60)

for model in MODELS:
    accs = []
    ch_accs = []
    params = None
    for seed in SEEDS:
        mpath = BASE / f"{model}_seed{seed}" / "metrics.json"
        if mpath.exists():
            d = json.loads(mpath.read_text())
            accs.append(d.get("acc_final", 0))
            ch_accs.append(d.get("changed_acc", 0))
            if params is None:
                params = d.get("params", "?")
    
    if len(accs) >= 3:
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        ch_mean = np.mean(ch_accs)
        ch_std = np.std(ch_accs)
        print(f"\n  {model} ({params} params)")
        print(f"    acc:      {acc_mean:.4f} +/- {acc_std:.4f}")
        print(f"    ch_acc:   {ch_mean:.4f} +/- {ch_std:.4f}")
        print(f"    seeds:    {[f'{a:.4f}' for a in accs]}")
        print(f"    range:    [{min(accs):.4f}, {max(accs):.4f}]")
    else:
        print(f"\n  {model}: only {len(accs)}/{len(SEEDS)} results found")

print("\nDone!")
