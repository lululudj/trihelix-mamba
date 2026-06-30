#!/bin/bash
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_multiseed

SEEDS=(42 123 456 789 1024)
MODELS=("single_chain" "transformer" "three_chain")
MAX_STEPS=3000
BATCH_SIZE=32
TOTAL=15
SKIPPED=0

echo "=== Multi-Seed Benchmark ==="
echo "Started: $(date)"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NAME="${model}_seed${seed}"
    OUT_DIR="results_wsl/benchmark_multiseed/${NAME}"
    
    if [ -f "${OUT_DIR}/metrics.json" ]; then
      ACC=$(python3 -c "import json; print(f\"{json.load(open('${OUT_DIR}/metrics.json')).get('acc_final',0):.4f}\")" 2>/dev/null || echo "?")
      echo "[SKIP] ${NAME} (acc=${ACC})"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    
    echo ""
    echo "=== [${model} seed=${seed}] $(date) ==="
    python3 train.py --model "${model}" --seed "${seed}" --max_steps "${MAX_STEPS}" --batch_size "${BATCH_SIZE}" --eval_every 500 --log_every 100 --out_dir "${OUT_DIR}"
    
    if [ -f "${OUT_DIR}/best.pt" ]; then
      python3 eval.py --checkpoint "${OUT_DIR}/best.pt" --data_root data_split --out "${OUT_DIR}/metrics.json"
      ACC=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/metrics.json')); print(f\"{d.get('acc_final',0):.4f}\")" 2>/dev/null || echo "?")
      echo "  => acc=${ACC}"
    fi
  done
done

echo ""
echo "=== ALL DONE: $(date) ==="
