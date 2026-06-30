#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Eval ==="
python3 eval.py --checkpoint results_wsl/run_hetero_v4_seed0/best.pt --data_root data_split --out results_wsl/run_hetero_v4_seed0/metrics.json
echo ""
echo "=== OOD ==="
python3 eval_ood.py --checkpoint results_wsl/run_hetero_v4_seed0/best.pt --out results_wsl/run_hetero_v4_seed0/ood_metrics.json
echo "=== Done ==="
cat results_wsl/run_hetero_v4_seed0/metrics.json
