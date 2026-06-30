import json, glob, os
from pathlib import Path

scaled_dir = Path("/mnt/e/three_chain_v3/results_wsl/benchmark_scaled")
ablation_dir = Path("/mnt/e/three_chain_v3/results_wsl/ablation")

# Count completed runs
scaled_done = []
scaled_trained = []
for d in scaled_dir.iterdir():
    if d.is_dir():
        if (d / "metrics.json").exists():
            scaled_done.append(d.name)
        elif (d / "best.pt").exists():
            scaled_trained.append(d.name)

abl_done = []
for d in ablation_dir.iterdir():
    if d.is_dir() and (d / "metrics.json").exists():
        abl_done.append(d.name)

# Find active training
import subprocess
res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
active = []
for line in res.stdout.split("\n"):
    if "python3 train.py" in line and "grep" not in line:
        parts = line.split()
        out_dir = ""
        for i, p in enumerate(parts):
            if p == "--out_dir" and i+1 < len(parts):
                out_dir = parts[i+1].split("/")[-1]
        if out_dir:
            # Get step
            log = Path(scaled_dir) / out_dir / "log.jsonl"
            if log.exists():
                lines = [l for l in open(log).readlines() if '"step"' in l and '"elapsed"' in l]
                if lines:
                    last = json.loads(lines[-1])
                    active.append(f"{out_dir} step={last['step']}/3000 elapsed={last['elapsed']}s")

print(f"Scaled done: {len(scaled_done)}/10  {[n.replace('single_scaled_','') for n in scaled_done]}")
print(f"Scaled trained: {len(scaled_trained)}/10")
print(f"Ablation done: {len(abl_done)}/15")
if active:
    for a in active:
        print(f"Active: {a}")
else:
    print("No active training")
