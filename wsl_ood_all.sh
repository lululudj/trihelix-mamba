#!/bin/bash
# OOD eval for all existing benchmark checkpoints - FIXED
cd /mnt/e/three_chain_v3

MODELS=("three_chain" "single_chain" "transformer")
SEEDS=(42 123 456 789 1024)

echo "=== OOD Eval: All Checkpoints ==="
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NAME="${model}_seed${seed}"
    CKPT="results_wsl/benchmark_multiseed/${NAME}/best.pt"
    OUT="results_wsl/benchmark_multiseed/${NAME}/ood_metrics.json"
    if [ -f "${OUT}" ]; then
      echo "[SKIP] ${NAME}"
      continue
    fi
    if [ -f "${CKPT}" ]; then
      echo "OOD eval: ${NAME}"
      python3 eval_ood.py --checkpoint "${CKPT}" --data_root data/ood_T150 --out "${OUT}"
    fi
  done
done
echo "=== OOD Done ==="
