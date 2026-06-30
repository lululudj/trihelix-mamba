#!/bin/bash
# ThreeChainMamba2 (A版本) 5种子 benchmark
# 8GB 显存约束: batch_size=4, expandable_segments
#
# 用法:
#   bash wsl_bench_mamba2.sh              # 全部5种子
#   bash wsl_bench_mamba2.sh 42           # 只跑 seed=42
#   bash wsl_bench_mamba2.sh 42 5000      # seed=42, 5000步

set -e

MODEL=three_chain_mamba2
CONFIG=configs/matched_mamba2.yaml
OUT_BASE=results_wsl/benchmark_mamba2
DATA_ROOT=./data_split
OOD_ROOT=./data/ood_T150

# 参数: 种子列表(默认5种子) + max_steps(默认5000)
SEEDS="${1:-}"
MAX_STEPS="${2:-5000}"
BATCH_SIZE=4
EVAL_EVERY=500

if [ -z "$SEEDS" ]; then
    SEEDS="42 123 456 789 1024"
fi

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ThreeChainMamba2 (A) Benchmark ==="
echo "seeds: $SEEDS"
echo "max_steps: $MAX_STEPS"
echo "batch_size: $BATCH_SIZE"
echo ""

for seed in $SEEDS; do
    OUT_DIR=$OUT_BASE/${MODEL}_seed${seed}
    echo ">>> seed=$seed → $OUT_DIR"

    # 训练
    python3 train.py --model $MODEL --config $CONFIG --seed $seed \
        --max_steps $MAX_STEPS --batch_size $BATCH_SIZE \
        --out_dir $OUT_DIR --data_root $DATA_ROOT \
        --eval_every $EVAL_EVERY --log_every 100 \
        --save_every $MAX_STEPS

    # OOD 评估(用 best.pt, 如果没有用 final.pt)
    CKPT=$OUT_DIR/best.pt
    if [ ! -f "$CKPT" ]; then
        CKPT=$OUT_DIR/final.pt
    fi
    if [ -f "$CKPT" ]; then
        python3 eval_ood.py --checkpoint $CKPT --config $CONFIG \
            --data_root $OOD_ROOT --out $OUT_DIR/ood_metrics.json \
            --batch_size 8
        echo "  OOD 评估完成 → $OUT_DIR/ood_metrics.json"
    else
        echo "  ⚠️ 未找到 checkpoint, 跳过 OOD 评估"
    fi
    echo ""
done

echo "=== Benchmark 完成 ==="
echo "结果收集: python3 collect_ood_mamba2.py"
