#!/bin/bash
# 消融实验第1轮: baseline vs +自反分身(碱基对)
# 同样 seed=42, 100 步, batch=4, 严格对照
cd /mnt/e/three_chain_v3
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "############################################################"
echo "#  消融实验第1轮: 验证'自反分身'配对思路有没有用"
echo "#  A = baseline (three_chain_mamba2, 无配对)"
echo "#  B = treatment (three_chain_mamba2_bp, +自反分身)"
echo "############################################################"

for MODEL in three_chain_mamba2 three_chain_mamba2_bp; do
    echo ""
    echo "============================================================"
    echo ">>> 跑 $MODEL ..."
    echo "============================================================"
    python3 probe_mamba2_brain.py \
        --model $MODEL \
        --config configs/matched_mamba2.yaml \
        --max_steps 100 \
        --seed 42 \
        --batch_size 4 2>&1 | tail -45
    echo "--- $MODEL 完成 ---"
done

echo ""
echo "============================================================"
echo ">>> 对照实验完成, 请对比两者的 zero_ratio / changed_acc / OOD非零预测数"
echo "============================================================"
