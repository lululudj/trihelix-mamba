#!/bin/bash
# setup/gen_data.sh — 生成 GridWorld 训练数据
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/gen_data.sh
#
# 调用 data/gen_grid_world.py 生成训练数据到 data_split/
# OOD 测试数据 data/ood_T150/ 已含在仓库里, 无需重新生成
#
# 估计耗时: ~30 秒

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== 生成 GridWorld 训练数据 ==="
echo "项目根: $PROJECT_ROOT"
echo

# 已存在则跳过
if [ -d data_split ] && [ "$(find data_split -name '*.npz' | wc -l)" -gt 0 ]; then
    N=$(find data_split -name '*.npz' | wc -l)
    echo "✅ data_split/ 已存在 ($N 个 .npz), 跳过生成"
    echo "   如需重新生成, 先删除: rm -rf data_split/"
    exit 0
fi

echo "调用 data/gen_grid_world.py..."
python data/gen_grid_world.py

# 校验
if [ ! -d data_split ]; then
    echo "❌ data_split/ 未生成"
    exit 1
fi
N=$(find data_split -name '*.npz' | wc -l)
if [ "$N" -eq 0 ]; then
    echo "❌ data_split/ 为空, 生成失败"
    exit 1
fi

echo
echo "✅ 训练数据已生成到 data_split/ ($N 个 .npz 文件)"
echo
echo "下一步:"
echo "  bash setup/train_main.sh    # 从零训练 3.26M 主力模型"
echo "  bash setup/verify_gate.sh   # 非退化门验证"
