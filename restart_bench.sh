#!/bin/bash
rm -rf /mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s123
rm -rf /mnt/e/three_chain_v3/results_wsl/benchmark_scaled/single_scaled_s456
python3 -c "import torch; torch.cuda.empty_cache(); print('ready')"
echo "cleanup done"
