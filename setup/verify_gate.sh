#!/bin/bash
# setup/verify_gate.sh — 非退化门验证 (论文基准准入门槛)
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/verify_gate.sh
#
# 调用 probe_mamba2_brain.py: 训练 2000 步 + OOD 诊断 + 非退化门判定
# 退出码 0 = PASS (通过非退化门), 1 = FAIL (退化)
#
# 非退化门门槛 (来自 README.md):
#   zero_ratio < 0.90  (不能全猜 0)
#   changed_acc > 0.30 (比随机 1/16 = 0.0625 强)
#   OOD 非零预测 > 目标的 5%
#
# 估计耗时: RTX 4060 ~10-15 分钟 (训练 + 评估)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== 非退化门验证 (probe_mamba2_brain.py) ==="
echo "项目根: $PROJECT_ROOT"
echo

# 确保训练数据
if [ ! -d data_split ] || [ "$(find data_split -name '*.npz' | wc -l)" -eq 0 ]; then
    echo "训练数据不存在, 调用 gen_data.sh..."
    bash setup/gen_data.sh
fi

# 确认 OOD 数据
if [ ! -d data/ood_T150 ] || [ "$(find data/ood_T150 -name '*.npz' | wc -l)" -eq 0 ]; then
    echo "❌ OOD 测试数据 data/ood_T150/ 不存在或为空"
    echo "   该目录应已含在仓库里 (72 样本 T=150)"
    exit 1
fi

echo "运行 probe_mamba2_brain.py (训练 + OOD 诊断 + 非退化门判定)..."
echo "  模型: three_chain_mamba2 (3.26M)"
echo "  max_steps: 2000, seed: 42, batch_size: 4"
echo "  估计耗时: RTX 4060 ~10-15 分钟"
echo

# probe_mamba2_brain.py 退出码: 0 = PASS, 1 = FAIL
set +e
python probe_mamba2_brain.py \
    --model three_chain_mamba2 \
    --config configs/matched_mamba2.yaml \
    --max_steps 2000 \
    --seed 42 \
    --batch_size 4 2>&1 | tee /tmp/gate_output.txt
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo
echo "=== 验证结果 ==="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "✅ 非退化门通过 (probe 退出码 0)"
    echo "   模型未退化成全猜 0, 可以进入正式 benchmark"
    echo
    echo "下一步:"
    echo "  bash setup/run_5seeds.sh      # 5-seed 完整基准"
    echo "  bash setup/run_ablations.sh    # 消融实验"
    exit 0
else
    echo "❌ 非退化门未通过 (probe 退出码 $EXIT_CODE)"
    echo "   模型可能退化, 检查上方输出确认原因"
    echo "   可能原因:"
    echo "     1. 训练步数不足 (2000 步可能不够, 试 5000)"
    echo "     2. 学习率不对 (检查 configs/matched_mamba2.yaml)"
    echo "     3. 数据问题 (检查 data_split/ 内容)"
    exit 1
fi
