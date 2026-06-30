#!/bin/bash
# ============================================
# Multi-Seed Professional Benchmark Suite
# 5 seeds ? 3 models = 15 training runs
# ============================================

cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_multiseed

SEEDS=(42 123 456 789 1024)
MODELS=("three_chain" "single_chain" "transformer")
MAX_STEPS=3000
BATCH_SIZE=32
TOTAL=$(( ${#SEEDS[@]} * ${#MODELS[@]} ))
CURRENT=0

echo "============================================"
echo "  Multi-Seed Benchmark: $TOTAL total runs"
echo "  Started: $(date)"
echo "============================================"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    NAME="${model}_seed${seed}"
    OUT_DIR="results_wsl/benchmark_multiseed/${NAME}"
    
    echo ""
    echo "[${CURRENT}/${TOTAL}] Training ${NAME}..."
    echo "  Model: ${model} | Seed: ${seed} | Steps: ${MAX_STEPS}"
    echo "  Started: $(date)"
    
    START_TIME=$SECONDS
    
    python3 train.py \
      --model "${model}" \
      --seed "${seed}" \
      --max_steps "${MAX_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --eval_every 500 \
      --log_every 100 \
      --out_dir "${OUT_DIR}" 2>&1 | tail -5
    
    ELAPSED=$((SECONDS - START_TIME))
    echo "  Finished: $(date) | Elapsed: ${ELAPSED}s"
    
    # Quick eval
    if [ -f "${OUT_DIR}/best.pt" ]; then
      echo "  Evaluating ${NAME}..."
      python3 eval.py \
        --checkpoint "${OUT_DIR}/best.pt" \
        --data_root data_split \
        --out "${OUT_DIR}/metrics.json" 2>&1 | grep -E "acc_final|changed_acc|params"
    fi
    
  done
done

echo ""
echo "============================================"
echo "  All $TOTAL runs complete!"
echo "  Finished: $(date)"
echo "============================================"

# Collect summary
echo ""
echo "=== SUMMARY ==="
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NAME="${model}_seed${seed}"
    METRICS="results_wsl/benchmark_multiseed/${NAME}/metrics.json"
    if [ -f "${METRICS}" ]; then
      ACC=$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d.get('acc_final',0):.4f}\")")
      CH=$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d.get('changed_acc',0):.4f}\")")
      echo "  ${NAME}: acc=${ACC}  ch_acc=${CH}"
    fi
  done
done
