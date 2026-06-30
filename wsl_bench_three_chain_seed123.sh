#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Benchmark: three_chain_seed123 ==="
echo "Started: $(date)"
python3 train.py --model three_chain --seed 123 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/benchmark_multiseed/three_chain_seed123
echo "Finished: $(date)"
