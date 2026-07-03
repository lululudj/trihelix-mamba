#!/bin/bash
# 官方 Mamba3 100 步对照训练 (与 mamba3_ref / mamba2 公平对照)
# 用 PYTHONPATH dev mode 加载 /mnt/e/mamba_official 的 mamba_ssm 2.3.2.post1 (含 Mamba3 + Triton)
# 首步含 Triton JIT 编译 ~55s, 后续 99 步 ~6s, 预计总时长 ~65s
source "$HOME/mamba3_venv/bin/activate"
cd /mnt/e/three_chain_v3
export PYTHONPATH=/mnt/e/mamba_official:$PYTHONPATH
# 显存碎片化预防 (memory 约束: GPU 8GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train.py \
  --model three_chain_mamba3 \
  --config configs/matched_mamba3.yaml \
  --max_steps 100 \
  --batch_size 2 \
  --data_root ./data_split_m3 \
  --out_dir results/run_three_chain_mamba3_official_seed0 \
  2>&1 | tee /mnt/e/three_chain_v3/results/run_three_chain_mamba3_official_seed0_train.log
