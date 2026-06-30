#!/bin/bash
cd /mnt/e/three_chain_v3
CKPT="results_wsl/run_hetero_v4_seed0/best.pt"
echo "=== Exp1 ==="
python3 killer_experiments/exp1_chain_ablation/run_exp1.py --checkpoint $CKPT --data_root data_split --out_dir results_wsl/exp1_v4
echo "=== Exp2 ==="
python3 killer_experiments/exp2_causal_needle/run_exp2.py --checkpoint $CKPT --num_sequences 5 --steps_per_seq 1000 --out_dir results_wsl/exp2_v4
echo "=== Done ==="
