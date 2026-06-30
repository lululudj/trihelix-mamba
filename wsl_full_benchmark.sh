#!/bin/bash
cd /mnt/e/three_chain_v3
mkdir -p results_wsl/benchmark_multiseed results_wsl/benchmark_scaled

echo "============================================"
echo "  PHASE 1: OOD Eval (15 checkpoints)"
echo "============================================"
bash wsl_ood_all.sh

echo ""
echo "============================================"
echo "  PHASE 2: Scaled Baselines (10 runs)"
echo "============================================"
bash wsl_benchmark_scaled.sh

echo ""
echo "============================================"
echo "  ALL DONE: $(date)"
echo "============================================"
