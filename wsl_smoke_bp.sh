#!/bin/bash
# ThreeChainBP 烟雾测试：真实 Mamba + GPU，100 步
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== ThreeChainBP 烟雾测试（碱基对耦合）$(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo ""

timeout 300 python3 train.py \
    --model three_chain_bp \
    --max_steps 100 \
    --eval_every 50 \
    --out_dir results_wsl_smoke/run_bp_smoke \
    2>&1 | tee smoke_bp.log

echo ""
echo "=== 烟雾测试完成 $(date) ==="
echo "=== log ==="
cat results_wsl_smoke/run_bp_smoke/log.jsonl 2>/dev/null | head -10
