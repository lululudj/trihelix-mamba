#!/bin/bash
# WSL 三模型正式训练：真实 Mamba + GPU，3000 步
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== WSL 三模型训练开始 $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"
echo ""

# 清理旧的 WSL 结果（如果有）
rm -rf results_wsl
mkdir -p results_wsl

start=$(date +%s)

for model in three_chain single_chain transformer; do
    echo ""
    echo "[$(date +%H:%M:%S)] >>> 开始训练 $model"
    m_start=$(date +%s)
    python3 train.py \
        --model "$model" \
        --seed 0 \
        --max_steps 3000 \
        --eval_every 500 \
        --out_dir "results_wsl/run_${model}_seed0" \
        2>&1 | tee "results_wsl/train_${model}.log"
    m_end=$(date +%s)
    echo "[$(date +%H:%M:%S)] <<< $model 完成 用时 $((m_end - m_start))s"
done

end=$(date +%s)
echo ""
echo "=== WSL 三模型训练全部完成 $(date) ==="
echo "总用时: $((end - start))s"
echo ""
echo "=== 训练 summary ==="
for model in three_chain single_chain transformer; do
    echo "--- $model ---"
    cat "results_wsl/run_${model}_seed0/summary.json" 2>/dev/null
    echo ""
done
