#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== V6 CTM+JEPA Training ==="
echo "Started: 06/29/2026 15:46:24"
python3 train.py --model three_chain --seed 0 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 50 --out_dir results_wsl/run_hetero_v6_seed0
echo "Finished: 06/29/2026 15:46:24"
