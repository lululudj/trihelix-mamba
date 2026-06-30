#!/bin/bash
# 冒烟测试：ThreeChainBP v2 能否构建+前向+反向（2 步训练）
set -e
cd "/mnt/e/新建文件夹 (2)/three_chain_validation"
python3 train.py \
    --model three_chain_bp \
    --seed 0 \
    --max_steps 2 \
    --eval_every 9999 \
    --out_dir results_wsl/_smoke_bp_v2 \
    2>&1 | tail -15
echo ""
echo "=== 验证 BP 模块参数 ==="
python3 -c "
import sys; sys.path.insert(0, '.')
from utils import load_config, build_model
cfg = load_config('configs/default.yaml')
m = build_model('three_chain_bp', cfg)
bp_params = [n for n, _ in m.named_parameters() if n.startswith('bp.')]
print(f'BP 参数数量: {len(bp_params)}')
# 确认没有 gate 参数（v2 已移除）
gate_params = [n for n in bp_params if 'gate' in n]
print(f'gate 参数（应为空）: {gate_params}')
print(f'BP 模块列表: {[n.split(\".\")[1] for n in bp_params[:8]]}...')
"
