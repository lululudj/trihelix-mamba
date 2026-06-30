#!/bin/bash
# ============================================
#  PROFESSIONAL BENCHMARK v2 (fixed)
# ============================================
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_scaled results_wsl/ablation

SEEDS=(42 123 456 789 1024)
SEEDS_SMALL=(42 123 456)
MAX_STEPS=3000
BATCH=32

echo "============================================"
echo "  PHASE A: Scaled Baselines"
echo "  Started: $(date)"
echo "============================================"

# --- SingleChain scaled (n_layers=10) ---
for seed in "${SEEDS[@]}"; do
  NAME="single_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  CFG="configs/scaled_single.yaml"
  if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  echo "=== [${NAME}] ==="
  python3 train.py --model single_chain --config ${CFG} --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED: ${NAME}"
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED: ${NAME}"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED: ${NAME}"
  fi
done

# --- Transformer scaled (n_layers=6) ---
for seed in "${SEEDS[@]}"; do
  NAME="trans_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  CFG="configs/scaled_transformer.yaml"
  if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  echo "=== [${NAME}] ==="
  python3 train.py --model transformer --config ${CFG} --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED: ${NAME}"
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED: ${NAME}"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED: ${NAME}"
  fi
done

echo ""
echo "============================================"
echo "  PHASE B: Ablation Matrix"
echo "  Started: $(date)"
echo "============================================"

ABL_MODELS=(three_chain three_chain_no_eagle three_chain_no_bind concat_mamba single_chain)
ABL_LABELS=(full no_eagle no_bind concat single)

for i in "${!ABL_MODELS[@]}"; do
  model="${ABL_MODELS[$i]}"
  label="${ABL_LABELS[$i]}"
  for seed in "${SEEDS_SMALL[@]}"; do
    NAME="abl_${label}_s${seed}"
    OUT="results_wsl/ablation/${NAME}"
    if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
    echo "=== [${NAME}] ==="
    python3 train.py --model ${model} --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED: ${NAME}"
    if [ -f "${OUT}/best.pt" ]; then
      python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED: ${NAME}"
      python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED: ${NAME}"
    fi
  done
done

echo ""
echo "=== ALL DONE: $(date) ==="
