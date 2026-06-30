#!/bin/bash
# 跑剩余 4 个种子 (123 456 789 1024)
cd /mnt/e/three_chain_v3
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in 123 456 789 1024; do
    echo ">>> seed=$seed"
    OUT_DIR=results_wsl/benchmark_mamba2/three_chain_mamba2_seed${seed}
    python3 train.py --model three_chain_mamba2 --config configs/matched_mamba2.yaml \
        --seed $seed --max_steps 2000 --batch_size 4 \
        --out_dir $OUT_DIR --data_root ./data_split \
        --eval_every 1000 --log_every 200 --save_every 2000 2>&1 | tail -5
    CKPT=$OUT_DIR/best.pt
    if [ ! -f "$CKPT" ]; then
        CKPT=$OUT_DIR/final.pt
    fi
    python3 eval_ood.py --checkpoint $CKPT --config configs/matched_mamba2.yaml \
        --data_root ./data/ood_T150 --out $OUT_DIR/ood_metrics.json \
        --batch_size 4 2>&1 | tail -8
    echo "---"
done
echo "=== 4 seeds done ==="
