#!/bin/bash
cd /mnt/e/three_chain_v3
CKPT="results_wsl/run_hetero_v4_seed0/best.pt"
python3 killer_experiments/exp1_chain_ablation/run_exp1.py --checkpoint $CKPT --data_root data_split --out_dir results_wsl/exp1_v4
