#!/bin/bash
# ThreeChainBP 正式训练 + 评估 + OOD
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== ThreeChainBP 正式训练开始 $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# 训练 3000 步
python3 train.py \
    --model three_chain_bp \
    --seed 0 \
    --max_steps 3000 \
    --eval_every 500 \
    --out_dir results_wsl/run_three_chain_bp_seed0 \
    2>&1 | tee results_wsl/train_three_chain_bp.log

echo ""
echo "=== 训练完成 $(date) ==="
cat results_wsl/run_three_chain_bp_seed0/summary.json
echo ""

# test 评估
echo "=== test 评估 ==="
python3 eval.py \
    --checkpoint results_wsl/run_three_chain_bp_seed0/best.pt \
    --out results_wsl/run_three_chain_bp_seed0/metrics.json \
    --plot 2>&1 | tee results_wsl/eval_bp.log

# OOD 评估
echo ""
echo "=== OOD T=150 评估 ==="
python3 eval_ood.py \
    --checkpoint results_wsl/run_three_chain_bp_seed0/best.pt \
    --data_root ./data/ood_T150 \
    --out results_wsl/run_three_chain_bp_seed0/ood_metrics.json 2>&1 | tee results_wsl/eval_ood_bp.log

echo ""
echo "=== 全部完成 $(date) ==="
