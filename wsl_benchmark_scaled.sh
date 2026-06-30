#!/bin/bash
# ============================================
# Parameter-Matched Baseline Benchmark
# SingleChain n_layers=10 (4.65M) + Transformer n_layers=6 (5.01M)
# 5 seeds each + OOD eval
# ============================================
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_scaled

SEEDS=(42 123 456 789 1024)
MAX_STEPS=3000
BATCH_SIZE=32

echo "=== Scaled Baseline Benchmark ==="
echo "Started: $(date)"

# ---- SingleChain scaled (n_layers=10) ----
for seed in "${SEEDS[@]}"; do
  NAME="single_scaled_seed${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/metrics.json" ]; then
    echo "[SKIP] ${NAME}"
    continue
  fi
  echo ""
  echo "=== [${NAME}] $(date) ==="
  python3 train.py --model single_chain --config configs/scaled_single.yaml --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH_SIZE} --eval_every 500 --log_every 100 --out_dir ${OUT}
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/ood_metrics.json"
  fi
done

# ---- Transformer scaled (n_layers=6) ----
for seed in "${SEEDS[@]}"; do
  NAME="transformer_scaled_seed${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/metrics.json" ]; then
    echo "[SKIP] ${NAME}"
    continue
  fi
  echo ""
  echo "=== [${NAME}] $(date) ==="
  python3 train.py --model transformer --config configs/scaled_transformer.yaml --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH_SIZE} --eval_every 500 --log_every 100 --out_dir ${OUT}
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/ood_metrics.json"
  fi
done

echo ""
echo "=== Scaled Benchmark Done: $(date) ==="
