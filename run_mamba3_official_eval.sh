#!/bin/bash
# 官方 Mamba3 OOD 评估 (与 mamba3_ref / mamba2 公平对照)
# 用 PYTHONPATH dev mode 加载 /mnt/e/mamba_official 的 mamba_ssm 2.3.2.post1
source "$HOME/mamba3_venv/bin/activate"
cd /mnt/e/three_chain_v3
export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 eval_ood.py \
  --checkpoint results/run_three_chain_mamba3_official_seed0/final.pt \
  --config configs/matched_mamba3.yaml \
  --data_root ./data/ood_T150 \
  --batch_size 4 \
  --out results/run_three_chain_mamba3_official_seed0/ood_metrics.json \
  2>&1 | tee results/run_three_chain_mamba3_official_seed0_eval.log
