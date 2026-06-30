#!/bin/bash
cd /mnt/e/three_chain_v3
CKPT="results_wsl/run_hetero_v5_seed0/best.pt"
python3 killer_experiments/exp3_concept_drift/run_exp3.py --checkpoint $CKPT --N 8 --K 4 --T 100 --num_sequences 30 --out_dir results_wsl/exp3_v5
