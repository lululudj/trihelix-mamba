import json, os
from pathlib import Path

base = Path("/mnt/e/three_chain_v3/results_wsl")

# Multi-seed standard benchmark
print("=" * 60)
print("STANDARD BENCHMARK (multi-seed, 3000 steps)")
print("=" * 60)
multi = base / "benchmark_multiseed"
for d in sorted(multi.iterdir()):
    m = d / "metrics.json"
    o = d / "ood_metrics.json"
    if m.exists() and o.exists():
        md = json.load(open(m))
        od = json.load(open(o))
        print(f"{d.name:30s} | acc={md['acc_final']:.4f} | OOD={od['acc_final']:.4f}")

# Ablation
print()
print("=" * 60)
print("ABLATION (2000 steps)")
print("=" * 60)
abl = base / "ablation"
for d in sorted(abl.iterdir()):
    m = d / "metrics.json"
    o = d / "ood_metrics.json"
    if m.exists() and o.exists():
        md = json.load(open(m))
        od = json.load(open(o))
        print(f"{d.name:30s} | acc={md['acc_final']:.4f} | OOD={od['acc_final']:.4f}")

# Scaled
print()
print("=" * 60)
print("SCALED BASELINE (2000 steps)")
print("=" * 60)
scaled = base / "benchmark_scaled"
for d in sorted(scaled.iterdir()):
    m = d / "metrics.json"
    o = d / "ood_metrics.json"
    if m.exists() and o.exists():
        md = json.load(open(m))
        od = json.load(open(o))
        print(f"{d.name:30s} | acc={md['acc_final']:.4f} | OOD={od['acc_final']:.4f}")
