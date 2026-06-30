#!/usr/bin/env python3
"""Monitor training progress and auto-generate report when done."""
import json, glob, time, subprocess, os
from datetime import datetime

SCALED = "/mnt/e/three_chain_v3/results_wsl/benchmark_scaled"
ABL = "/mnt/e/three_chain_v3/results_wsl/ablation"

TARGET_SCALED = 10
TARGET_ABL = 15

print(f"Monitor started: {datetime.now().strftime('%H:%M:%S')}")
print(f"Targets: {TARGET_SCALED} scaled + {TARGET_ABL} ablation")

while True:
    scaled_done = len(glob.glob(f"{SCALED}/*/metrics.json"))
    abl_done = len(glob.glob(f"{ABL}/*/metrics.json"))
    
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Scaled: {scaled_done}/{TARGET_SCALED} | Ablation: {abl_done}/{TARGET_ABL}")
    
    if scaled_done >= TARGET_SCALED and abl_done >= TARGET_ABL:
        print("\nALL DONE! Generating report...")
        os.system("cd /mnt/e/three_chain_v3 && python3 generate_report.py")
        print("Report complete!")
        break
    
    time.sleep(600)  # Check every 10 minutes
