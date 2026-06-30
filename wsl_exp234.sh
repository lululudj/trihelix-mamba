#!/bin/bash
cd /mnt/e/three_chain_v3
CKPT="results_wsl/run_hetero_v5_seed0/best.pt"

echo "=== Exp2: Causal Needle (10 seq, 5000 steps)==="
python3 killer_experiments/exp2_causal_needle/run_exp2.py --num_sequences 10 --steps_per_seq 5000 --out_dir results_wsl/exp2_v5

echo ""
echo "=== Exp3: Concept Drift ==="
python3 killer_experiments/exp3_concept_drift/run_exp3.py --checkpoint $CKPT --N 8 --K 4 --T 100 --num_sequences 30 --out_dir results_wsl/exp3_v5

echo ""
echo "=== Exp4: Maze Navigation ==="
python3 killer_experiments/exp4_maze_navigation/run_exp4.py --maze_size 8 --explore_steps 200 --num_trials 10 --out_dir results_wsl/exp4_v5

echo ""
echo "=== ALL DONE ==="
