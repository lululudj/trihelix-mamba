#!/bin/bash
for d in single_scaled_s789 trans_scaled_s123 trans_scaled_s42 single_scaled_s1024; do
  path="/mnt/e/three_chain_v3/results_wsl/benchmark_scaled/$d"
  if [ -d "$path" ] && [ ! -f "$path/ood_metrics.json" ]; then
    rm -rf "$path"
    echo "removed zombie: $d"
  fi
done
echo "clean done"
