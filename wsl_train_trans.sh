#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Training transformer ==="
python3 train.py --model transformer --seed 0 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/run_transformer_v5_seed0
echo "=== Eval transformer ==="
python3 eval.py --checkpoint results_wsl/run_transformer_v5_seed0/best.pt --data_root data_split --out results_wsl/run_transformer_v5_seed0/metrics.json 2>&1 | grep -E "acc_final|changed_acc|params"
