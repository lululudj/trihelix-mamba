#!/bin/bash
# ============================================
#  PROFESSIONAL BENCHMARK: ALL IN ONE
#  Phase A: Scaled Baselines (10 runs)
#  Phase B: Ablation Matrix (5 variants x 3 seeds)
#  Phase C: OOD Eval on all new checkpoints
# ============================================
set -e
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_scaled results_wsl/ablation

SEEDS_SMALL=(42 123 456)
SEEDS=(42 123 456 789 1024)
MAX_STEPS=3000
BATCH=32
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "  PHASE A: Scaled Baselines (10 runs)"
echo "  Started: $(date)"
echo "============================================"

# --- SingleChain scaled (n_layers=10, ~4.65M) ---
for seed in "${SEEDS[@]}"; do
  NAME="single_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  echo "=== [${NAME}] $(date) ==="
  python3 train.py --model single_chain --config configs/scaled_single.yaml --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT}
  python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json"
  python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json"
done

# --- Transformer scaled (n_layers=6, ~5.01M) ---
for seed in "${SEEDS[@]}"; do
  NAME="trans_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  echo "=== [${NAME}] $(date) ==="
  python3 train.py --model transformer --config configs/scaled_transformer.yaml --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT}
  python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json"
  python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json"
done

echo ""
echo "============================================"
echo "  PHASE B: Ablation Matrix"
echo "  Started: $(date)"
echo "============================================"

ABLATIONS=("three_chain" "three_chain_no_eagle" "three_chain_no_bind" "concat_mamba" "single_chain")
ABL_LABELS=("full" "no_eagle" "no_bind" "concat" "single")

for i in "${!ABLATIONS[@]}"; do
  model="${ABLATIONS[$i]}"
  label="${ABL_LABELS[$i]}"
  for seed in "${SEEDS_SMALL[@]}"; do
    NAME="abl_${label}_s${seed}"
    OUT="results_wsl/ablation/${NAME}"
    if [ -f "${OUT}/metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
    echo "=== [${NAME}] model=${model} seed=${seed} ==="
    python3 train.py --model ${model} --seed ${seed} --max_steps ${MAX_STEPS} --batch_size ${BATCH} --eval_every 500 --log_every 100 --out_dir ${OUT}
    python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json"
  done
done

echo ""
echo "============================================"
echo "  ALL DONE: $(date)"
echo "============================================"
