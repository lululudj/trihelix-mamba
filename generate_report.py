#!/usr/bin/env python3
"""Professional Benchmark Report Generator
Collects all results and generates a complete evaluation report.
"""
import json, glob, sys, os
import numpy as np
from pathlib import Path

BASE = Path("/mnt/e/three_chain_v3/results_wsl")
BENCH = BASE / "benchmark_multiseed"
SCALED = BASE / "benchmark_scaled"
ABL = BASE / "ablation"
OUT = BASE / "benchmark_report.json"

SEEDS = [42, 123, 456, 789, 1024]
SEEDS_SMALL = [42, 123, 456]

def load_model_results(model_dir, seeds, prefix=""):
    """Load standard + OOD metrics for a model variant."""
    accs, ch_accs, ood_accs, ood_chs, params = [], [], [], [], None
    for s in seeds:
        mf = model_dir / f"{prefix}_seed{s}" / "metrics.json" if prefix else model_dir / f"{prefix}{s}" / "metrics.json"
        of = model_dir / (f"{prefix}_seed{s}" / "ood_metrics.json" if prefix else f"{prefix}{s}" / "ood_metrics.json")
        try:
            md = json.loads(mf.read_text())
            od = json.loads(of.read_text()) if of.exists() else {}
            accs.append(md.get("acc_final", 0))
            ch_accs.append(md.get("changed_acc", 0))
            ood_accs.append(od.get("acc_final", od.get("changed_acc", 0)))
            ood_chs.append(od.get("changed_acc", 0))
            if params is None: params = md.get("params", "?")
        except: pass
    return {
        "params": params,
        "n_seeds": len(accs),
        "acc_mean": float(np.mean(accs)) if accs else 0,
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0,
        "ch_acc_mean": float(np.mean(ch_accs)) if ch_accs else 0,
        "ch_acc_std": float(np.std(ch_accs, ddof=1)) if len(ch_accs) > 1 else 0,
        "ood_acc_mean": float(np.mean(ood_accs)) if ood_accs else 0,
        "ood_acc_std": float(np.std(ood_accs, ddof=1)) if len(ood_accs) > 1 else 0,
        "ood_ch_mean": float(np.mean(ood_chs)) if ood_chs else 0,
        "ood_ch_std": float(np.std(ood_chs, ddof=1)) if len(ood_chs) > 1 else 0,
        "accs": [float(a) for a in accs],
        "ood_accs": [float(a) for a in ood_accs],
    }

report = {}

# 1. Standard benchmark
report["standard"] = {
    "tri_helix": load_model_results(BENCH, SEEDS, "three_chain"),
    "single_mamba": load_model_results(BENCH, SEEDS, "single_chain"),
    "transformer": load_model_results(BENCH, SEEDS, "transformer"),
}

# 2. Scaled baselines (parameter-matched)
report["scaled"] = {
    "single_scaled": load_model_results(SCALED, SEEDS, "single_scaled"),
    "transformer_scaled": load_model_results(SCALED, SEEDS, "trans_scaled"),
}

# 3. Ablation matrix
ablation_labels = {
    "abl_full": "Tri-Helix (Full)",
    "abl_no_eagle": "No Eagle",
    "abl_no_bind": "No Bind",
    "abl_concat": "Concat Mamba",
    "abl_single": "Single Chain",
}
report["ablation"] = {}
for key, label in ablation_labels.items():
    report["ablation"][key] = load_model_results(ABL, SEEDS_SMALL, key)

# Save
json.dump(report, open(str(OUT), "w"), indent=2, ensure_ascii=False)
print(f"Report saved to {OUT}")

# Print summary
print("\n" + "=" * 70)
print("  PROFESSIONAL BENCHMARK REPORT")
print("=" * 70)

for section, models in [("STANDARD (5 seeds)", report["standard"]), 
                          ("SCALED PARAM-MATCHED (5 seeds)", report["scaled"]),
                          ("ABLATION (3 seeds)", report["ablation"])]:
    print(f"\n--- {section} ---")
    for name, r in models.items():
        if r["n_seeds"] > 0:
            print(f"  {name:25s}  params={str(r['params']):>8s}  "
                  f"acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
                  f"ood_ch={r['ood_ch_mean']:.4f}+/-{r['ood_ch_std']:.4f}  "
                  f"n={r['n_seeds']}")
        else:
            print(f"  {name:25s}  (no data)")

print("\nDone!")
