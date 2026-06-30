#!/bin/bash
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_scaled

SEEDS=(42 123 456 789 1024)
CFG="configs/scaled_single.yaml"

for seed in "${SEEDS[@]}"; do
  NAME="single_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  # Clean any partial previous run
  rm -rf "${OUT}"
  echo "=== [${NAME}] $(date) ==="
  python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
  python3 train.py --model single_chain --config ${CFG} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED: ${NAME}"
  python3 -c "import torch; torch.cuda.empty_cache()"
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --config ${CFG} --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED"
  fi
  python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared after run')"
done

echo "=== Single Scaled Done: $(date) ==="

# ---- Transformer scaled (n_layers=6) ----
CFG2="configs/scaled_transformer.yaml"
for seed in "${SEEDS[@]}"; do
  NAME="trans_scaled_s${seed}"
  OUT="results_wsl/benchmark_scaled/${NAME}"
  if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
  rm -rf "${OUT}"
  echo "=== [${NAME}] $(date) ==="
  python3 -c "import torch; torch.cuda.empty_cache()"
  python3 train.py --model transformer --config ${CFG2} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED"
  python3 -c "import torch; torch.cuda.empty_cache()"
  if [ -f "${OUT}/best.pt" ]; then
    python3 eval.py --checkpoint "${OUT}/best.pt" --config ${CFG2} --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED"
    python3 eval_ood.py --checkpoint "${OUT}/best.pt" --config ${CFG2} --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED"
  fi
  python3 -c "import torch; torch.cuda.empty_cache()"
done

echo "=== ALL SCALED DONE: $(date) ==="

# ---- Ablation ----
mkdir -p results_wsl/ablation
ABL_MODELS=(three_chain three_chain_no_eagle three_chain_no_bind concat_mamba single_chain)
ABL_LABELS=(full no_eagle no_bind concat single)
ABL_SEEDS=(42 123 456)

for i in "${!ABL_MODELS[@]}"; do
  for seed in "${ABL_SEEDS[@]}"; do
    NAME="abl_${ABL_LABELS[$i]}_s${seed}"
    OUT="results_wsl/ablation/${NAME}"
    if [ -f "${OUT}/ood_metrics.json" ]; then echo "[SKIP] ${NAME}"; continue; fi
    rm -rf "${OUT}"
    echo "=== [${NAME}] $(date) ==="
    python3 -c "import torch; torch.cuda.empty_cache()"
    python3 train.py --model ${ABL_MODELS[$i]} --seed ${seed} --max_steps 2000 --batch_size 16 --eval_every 500 --log_every 100 --out_dir ${OUT} || echo "TRAIN FAILED"
    python3 -c "import torch; torch.cuda.empty_cache()"
    if [ -f "${OUT}/best.pt" ]; then
      python3 eval.py --checkpoint "${OUT}/best.pt" --data_root data_split --out "${OUT}/metrics.json" || echo "EVAL FAILED"
      python3 eval_ood.py --checkpoint "${OUT}/best.pt" --data_root data/ood_T150 --out "${OUT}/ood_metrics.json" || echo "OOD FAILED"
    fi
    python3 -c "import torch; torch.cuda.empty_cache()"
  done
done

echo "=== ALL DONE: $(date) ==="
