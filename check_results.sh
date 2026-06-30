#!/bin/bash
for d in /mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s*/; do
  name=$(basename "$d")
  has_ood=""
  if [ -f "$d/ood_metrics.json" ]; then has_ood=" [OOD OK]"; fi
  has_best=""
  if [ -f "$d/best.pt" ]; then has_best=" [best.pt OK]"; fi
  echo "$name$has_best$has_ood"
done
echo "---"
for d in /mnt/e/three_chain_v3/results_wsl/benchmark_scaled/trans_scaled_s*/; do
  name=$(basename "$d")
  has_ood=""
  if [ -f "$d/ood_metrics.json" ]; then has_ood=" [OOD OK]"; fi
  echo "$name$has_ood"
done 2>/dev/null
echo "---ablation---"
ls /mnt/e/three_chain_v3/results_wsl/ablation/ 2>/dev/null
