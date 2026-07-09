#!/bin/bash
# setup/run_ablations.sh — 消融实验一键脚本
#
# 用法 (在 WSL Ubuntu 里, conda env mamba 已激活):
#   cd /mnt/e/three_chain_v3
#   bash setup/run_ablations.sh
#
# 跑 3 个消融对照 (来自 README.md / REPRODUCE.md):
#   1. BPv1 (three_chain_mamba2_bp): 链内碱基对 → OOD 应降 ~5.6%
#   2. BPv2 (three_chain_mamba2_bpv2): 跨链碱基对 → 应与 baseline 持平
#   3. Transformer-tiny: 同参数 Transformer → 应全猜 0 塌缩 (zero_ratio=1.0)
#
# 每个: probe_mamba2_brain.py 训练 2000 步 + OOD 诊断 + 非退化门判定
#
# 估计耗时: RTX 4060 ~45 分钟 (3 × ~15 分钟)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动激活 conda env mamba
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "mamba" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba
fi

echo "=== 消融实验 ==="
echo "项目根: $PROJECT_ROOT"
echo

# 确保训练数据
if [ ! -d data_split ] || [ "$(find data_split -name '*.npz' | wc -l)" -eq 0 ]; then
    echo "训练数据不存在, 调用 gen_data.sh..."
    bash setup/gen_data.sh
fi

MAX_STEPS=2000
SEED=42
BATCH=4
OUT_DIR_BASE="results_wsl/ablations"

run_ablation() {
    local NAME="$1"
    local MODEL="$2"
    local CONFIG="$3"
    local EXPECT="$4"

    local OUT_DIR="$OUT_DIR_BASE/$NAME"
    echo "=========================================="
    echo "  消融: $NAME"
    echo "  模型: $MODEL"
    echo "  配置: $CONFIG"
    echo "  期望: $EXPECT"
    echo "=========================================="

    # probe_mamba2_brain.py 退出码: 0=PASS, 1=FAIL
    set +e
    python probe_mamba2_brain.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --max_steps "$MAX_STEPS" \
        --seed "$SEED" \
        --batch_size "$BATCH" 2>&1 | tee "$OUT_DIR.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  非退化门: ✅ PASS"
    else
        echo "  非退化门: ❌ FAIL (可能退化)"
    fi
    echo
}

# 1. BPv1: 链内碱基对 (OOD 应降 ~5.6%)
run_ablation \
    "bpv1" \
    "three_chain_mamba2_bp" \
    "configs/matched_mamba2.yaml" \
    "OOD 降 ~5.6% (碱基对破坏外推)"

# 2. BPv2: 跨链碱基对 (应与 baseline 持平)
run_ablation \
    "bpv2" \
    "three_chain_mamba2_bpv2" \
    "configs/matched_mamba2.yaml" \
    "与 baseline 持平 (跨链不破坏)"

# 3. Transformer-tiny: 同参数 (应全猜 0 塌缩)
run_ablation \
    "transformer_tiny" \
    "transformer" \
    "configs/matched_transformer_tiny.yaml" \
    "zero_ratio=1.0 完全塌缩 (Transformer 无 AnchorInit2)"

echo
echo "=========================================="
echo "  消融实验完成"
echo "=========================================="
echo "期望对比 (README.md):"
echo "  baseline (three_chain_mamba2): changed_acc≈0.5952, decay≈0, PASS"
echo "  BPv1 (链内碱基对):              changed_acc 降 ~5.6%, 可能 PASS"
echo "  BPv2 (跨链碱基对):              与 baseline 持平, PASS"
echo "  Transformer-tiny:                zero_ratio=1.0, FAIL (塌缩)"
echo
echo "查看各实验日志: $OUT_DIR_BASE/*.log"
echo
echo "如需 5-seed 基准:"
echo "  bash setup/run_5seeds.sh"
