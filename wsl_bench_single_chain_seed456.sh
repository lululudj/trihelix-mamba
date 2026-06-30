#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Benchmark: single_chain_seed456 ==="
echo "Started: $(date)"
python3 train.py --model single_chain --seed 456 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/benchmark_multiseed/single_chain_seed456
echo "Finished: $(date)"
