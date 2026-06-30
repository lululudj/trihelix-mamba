#!/bin/bash
# WSL 评估：test 集 + OOD 长程外推 + aggregate
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"

echo "=== WSL 评估开始 $(date) ==="

MODELS="three_chain single_chain transformer"

# 1. test 集评估
echo ""
echo "=== [1/3] test 集评估 (T=100) ==="
for m in $MODELS; do
    echo ""
    echo "--- eval $m (test) ---"
    python3 eval.py \
        --checkpoint "results_wsl/run_${m}_seed0/best.pt" \
        --out "results_wsl/run_${m}_seed0/metrics.json" \
        --plot 2>&1 | tee -a "results_wsl/eval_${m}.log"
done

# 2. OOD 长程外推评估 (T=150)
echo ""
echo "=== [2/3] OOD 长程外推评估 (T=150) ==="
for m in $MODELS; do
    echo ""
    echo "--- eval $m (OOD T=150) ---"
    python3 eval_ood.py \
        --checkpoint "results_wsl/run_${m}_seed0/best.pt" \
        --data_root "./data/ood_T150" \
        --out "results_wsl/run_${m}_seed0/ood_metrics.json" 2>&1 | tee -a "results_wsl/eval_ood_${m}.log"
done

# 3. aggregate 汇总
echo ""
echo "=== [3/3] aggregate 汇总 ==="
python3 eval.py \
    --aggregate \
    results_wsl/run_three_chain_seed0/metrics.json \
    results_wsl/run_single_chain_seed0/metrics.json \
    results_wsl/run_transformer_seed0/metrics.json \
    --out results_wsl/summary.json 2>&1 | tee "results_wsl/aggregate.log"

echo ""
echo "=== 全部评估完成 $(date) ==="
