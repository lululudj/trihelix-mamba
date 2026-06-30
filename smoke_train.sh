#!/bin/bash
cd /mnt/e/three_chain_v3
python3 train.py --model three_chain --seed 0 --max_steps 10 --batch_size 16 --eval_every 100 --log_every 5 --out_dir results_wsl/smoke_v4
