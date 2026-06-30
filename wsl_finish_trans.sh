#!/bin/bash
cd /mnt/e/three_chain_v3
python3 train.py --model transformer --seed 0 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/run_transformer_v5_seed0 --resume results_wsl/run_transformer_v5_seed0/final.pt
echo "=== Eval ==="
python3 eval.py --checkpoint results_wsl/run_transformer_v5_seed0/best.pt --data_root data_split --out results_wsl/run_transformer_v5_seed0/metrics.json | grep -E "acc_final|changed_acc|params"
