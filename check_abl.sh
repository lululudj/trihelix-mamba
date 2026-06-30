#!/bin/bash
echo "processes: $(ps aux | grep 'train.py' | grep -v grep | wc -l)"
echo "---"
ls /mnt/e/three_chain_v3/results_wsl/ablation/ 2>/dev/null
echo "==="
for d in /mnt/e/three_chain_v3/results_wsl/ablation/*/; do
  n=$(basename "$d")
  if [ -f "${d}ood_metrics.json" ]; then
    echo "$n [DONE]"
  elif [ -f "${d}best.pt" ]; then
    echo "$n [training]"
  else
    echo "$n [empty]"
  fi
done
