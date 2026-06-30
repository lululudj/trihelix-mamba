#!/bin/bash
# WSL 烟雾测试：真实 Mamba + GPU，100 步
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== 烟雾测试：真实 Mamba + GPU (100 步) ==="
echo "时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo ""

# 烟雾测试 three_chain 100 步
timeout 300 python3 train.py \
    --model three_chain \
    --max_steps 100 \
    --eval_every 50 \
    --out_dir results_wsl_smoke/run_three_chain_smoke \
    2>&1 | tee smoke_wsl.log

echo ""
echo "=== 烟雾测试完成 $(date) ==="
echo "=== 检查 log ==="
cat results_wsl_smoke/run_three_chain_smoke/log.jsonl 2>/dev/null | head -20
