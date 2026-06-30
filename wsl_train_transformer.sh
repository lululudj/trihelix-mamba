#!/bin/bash
cd /mnt/e/three_chain_v3
echo "=== Training transformer baseline ==="
python3 train.py --model transformer --seed 0 --max_steps 3000 --batch_size 32 --eval_every 500 --log_every 100 --out_dir results_wsl/run_transformer_v5_seed0
echo "=== transformer done ==="
