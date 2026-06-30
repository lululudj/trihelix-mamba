#!/bin/bash
cd /mnt/e/three_chain_v3

echo "=== V4 HeteroMamba Training ==="
echo "Model: three_chain (HeteroMamba core)"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

python3 train.py \
    --model three_chain \
    --seed 0 \
    --max_steps 3000 \
    --batch_size 32 \
    --eval_every 500 \
    --log_every 50 \
    --out_dir results_wsl/run_hetero_v4_seed0

echo ""
echo "=== Training complete ==="
