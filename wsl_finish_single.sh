#!/bin/bash
cd /mnt/e/three_chain_v3
# Resume single_chain to finish the last few steps
python3 train.py --model single_chain --seed 0 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/run_single_v5_seed0 --resume results_wsl/run_single_v5_seed0/final.pt 2>&1 | tail -5
echo "=== Eval single_chain ==="
python3 eval.py --checkpoint results_wsl/run_single_v5_seed0/best.pt --data_root data_split --out results_wsl/run_single_v5_seed0/metrics.json 2>&1 | grep -E "acc_final|changed_acc"
