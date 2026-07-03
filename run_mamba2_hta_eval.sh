#!/bin/bash
# HTA 模型 OOD 长程外推评估 (白嫖测试)
# 不需 PYTHONPATH: HTA 用系统 mamba_ssm 2.2.4 (与 baseline 同)
set -e
cd /mnt/e/three_chain_v3

echo "=== OOD eval three_chain_mamba2_hta seed=0 ==="
python3 eval_ood.py \
  --checkpoint results/run_three_chain_mamba2_hta_seed0/final.pt \
  --config configs/matched_mamba2_hta.yaml \
  --data_root ./data/ood_T150 \
  --batch_size 4 \
  --seed 0 \
  --out results/run_three_chain_mamba2_hta_seed0/ood_metrics.json

echo "=== 完成 ==="
cat results/run_three_chain_mamba2_hta_seed0/ood_metrics.json
