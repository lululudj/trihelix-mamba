#!/bin/bash
# setup/train_main.sh — 从零训练 3.26M 主力模型
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/train_main.sh
#
# 流程:
#   1. 确保训练数据已生成 (调 gen_data.sh)
#   2. 跑 train.py 训练 three_chain_mamba2 (3.26M, seed=42, 2000 步)
#   3. 用训练出的 best.pt 跑 eval_ood.py 验证 OOD 不退化
#
# 估计耗时: RTX 4060 ~10-15 分钟
# 期望结果: 本机训练的 best.pt 跑 OOD, decay ≈ 0 (±2%), 与下载的 best.pt 一致

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== 从零训练 3.26M 主力模型 ==="
echo "项目根: $PROJECT_ROOT"
echo

# 1. 确保训练数据
if [ ! -d data_split ] || [ "$(find data_split -name '*.npz' | wc -l)" -eq 0 ]; then
    echo "训练数据不存在, 调用 gen_data.sh..."
    bash setup/gen_data.sh
fi

# 2. 训练
MODEL="three_chain_mamba2"
CONFIG="configs/matched_mamba2.yaml"
SEED=42
MAX_STEPS=2000
OUT_DIR="results/run_${MODEL}_seed${SEED}"

echo
echo "=== 训练 ==="
echo "  模型: $MODEL (3.26M params)"
echo "  配置: $CONFIG"
echo "  seed: $SEED, max_steps: $MAX_STEPS"
echo "  输出: $OUT_DIR/"
echo "  估计耗时: RTX 4060 ~10-15 分钟"
echo

python train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --out_dir "$OUT_DIR"

# 3. 校验 best.pt 已生成
TRAIN_BEST="$OUT_DIR/best.pt"
if [ ! -f "$TRAIN_BEST" ]; then
    echo "❌ 训练结束但 $TRAIN_BEST 未生成"
    echo "   可能训练过程中无 val 改善, 检查 $OUT_DIR/log.jsonl"
    exit 1
fi
echo
echo "✅ 训练完成, best.pt 已生成: $TRAIN_BEST"

# 4. 跑 OOD 验证
echo
echo "=== OOD 长程评估 (用本机训练的 best.pt) ==="
METRICS_JSON="$OUT_DIR/ood_metrics.json"
python eval_ood.py \
    --checkpoint "$TRAIN_BEST" \
    --config "$CONFIG" \
    --data_root ./data/ood_T150 \
    --out "$METRICS_JSON"

# 5. 打印关键指标
echo
echo "=== 本机训练结果 ==="
python <<EOF
import json
with open("$METRICS_JSON") as f:
    m = json.load(f)
ch = m.get("changed_acc", 0)
dec = m.get("ood_decay_pct")
print(f"  changed_acc (t=150): {ch:.4f}  (README 期望 ≈ 0.5952)")
if dec is not None:
    print(f"  ood_decay_pct: {dec*100:+.2f}%  (期望 ≈ 0%)")
if ch > 0.50 and (dec is None or abs(dec) < 0.05):
    print()
    print("✅ 本机训练验证通过: 模型未退化, 与下载的 best.pt 一致")
else:
    print()
    print("⚠️  本机训练结果偏离期望 (可能 seed 差异或训练步数不足)")
EOF

echo
echo "下一步:"
echo "  bash setup/verify_gate.sh     # 非退化门验证 (论文基准准入门槛)"
echo "  bash setup/run_5seeds.sh      # 5-seed 完整基准 (可选)"
