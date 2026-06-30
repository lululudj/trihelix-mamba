#!/bin/bash
cd /mnt/e/three_chain_v3
CKPT="results_wsl/run_hetero_v5_seed0/best.pt"
echo "=== V5 Eval ==="
python3 eval.py --checkpoint $CKPT --data_root data_split --out results_wsl/run_hetero_v5_seed0/metrics.json
echo ""
echo "=== V5 Exp1 ==="
python3 killer_experiments/exp1_chain_ablation/run_exp1.py --checkpoint $CKPT --data_root data_split --out_dir results_wsl/exp1_v5
echo ""
cat results_wsl/exp1_v5/exp1_results.json
