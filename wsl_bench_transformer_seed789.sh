#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Benchmark: transformer_seed789 ==="
echo "Started: $(date)"
python3 train.py --model transformer --seed 789 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/benchmark_multiseed/transformer_seed789
echo "Finished: $(date)"
