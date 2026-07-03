#!/bin/bash
# ThreeChainMamba2HTA (白嫖 Mamba3 heavy_tail_activation) 100 步训练
# 用系统 mamba_ssm 2.2.4 (和 baseline 一致, 不设 PYTHONPATH)
# 对比: baseline mamba2 (A=-exp) vs hta mamba2 (A=-heavy_tail), 仅 A 激活函数不同
cd /mnt/e/three_chain_v3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train.py \
  --model three_chain_mamba2_hta \
  --config configs/matched_mamba2_hta.yaml \
  --max_steps 100 \
  --batch_size 2 \
  --data_root ./data_split_m3 \
  --out_dir results/run_three_chain_mamba2_hta_seed0 \
  2>&1 | tee results/run_three_chain_mamba2_hta_seed0_train.log
